import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from src.models.utils import CartesianPositionalEmbedding

def is_square(n: float) -> bool:
    if n < 0:
        return False
    sqrt_n = math.sqrt(n)
    return sqrt_n ** 2 == n

class MultiHeadSTEVESA(ModelMixin, ConfigMixin):

    # enable diffusers style config and model save/load
    @register_to_config
    def __init__(self, num_iterations, num_slots, num_heads,
                 input_size, out_size, slot_size, mlp_hidden_size, 
                 input_resolution, epsilon=1e-8, 
                 learnable_slot_init=False, 
                 bi_level=False):
        super().__init__()

        self.pos = CartesianPositionalEmbedding(input_size, input_resolution)
        self.in_layer_norm = nn.LayerNorm(input_size)
        self.in_mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size)
            )
        if bi_level:
            assert learnable_slot_init, 'Bi-level training requires learnable_slot_init=True'

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.learnable_slot_init = learnable_slot_init
        self.bi_level = bi_level

        assert slot_size % num_heads == 0, 'slot_size must be divisible by num_heads'

        if learnable_slot_init:
            self.slot_mu = nn.Parameter(torch.Tensor(1, num_slots, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
        else:
            # parameters for Gaussian initialization (shared by all slots).
            self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # linear maps for the attention module.
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(input_size, slot_size, bias=False)
        self.project_v = nn.Linear(input_size, slot_size, bias=False)

        # slot update functions.
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size))
        
        self.out_layer_norm = nn.LayerNorm(slot_size)
        self.out_linear = nn.Linear(slot_size, out_size)
        
    def forward(self, inputs):
        slots_collect, attns_collect = self.forward_slots(inputs)
        slots_collect = self.out_layer_norm(slots_collect)
        slots_collect = self.out_linear(slots_collect)
        return slots_collect, attns_collect

    def forward_slots(self, inputs):
        """
        inputs: batch_size x seq_len x input_size x h x w
        return: batch_size x num_slots x slot_size
        """
        B, T, input_size, h, w = inputs.size()
        inputs = self.pos(inputs)
        inputs = rearrange(inputs, 'b t n_inp h w -> b t (h w) n_inp')
        inputs = self.in_mlp(self.in_layer_norm(inputs))

        # num_inputs = h * w

        if self.learnable_slot_init:
            slots = repeat(self.slot_mu, '1 num_s d -> b num_s d', b=B)
        else:
            # initialize slots
            slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
            slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = rearrange(self.project_k(inputs), 'b t n_inp (h d) -> b t h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, T, num_heads, num_inputs, slot_size].
        v = rearrange(self.project_v(inputs), 'b t n_inp (h d) -> b t h n_inp d',
                      h=self.num_heads)  # Shape: [batch_size, T, num_heads, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k

        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                if self.bi_level and i == self.num_iterations - 1:
                    slots = slots.detach() + self.slot_mu - self.slot_mu.detach()
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = rearrange(self.project_q(slots), 'b n_s (h d) -> b h n_s d',
                              h=self.num_heads)  # Shape: [batch_size, num_heads, num_slots, slot_size].
                attn_logits = torch.einsum('...id,...sd->...is', k[:, t],
                                           q)  # Shape: [batch_size, num_heads, num_inputs, num_slots]
                attn = F.softmax(rearrange(attn_logits, 'b h n_inp n_s -> b n_inp (h n_s)'), -1)
                attn_vis = rearrange(attn, 'b n_inp (h n_s) -> b h n_inp n_s', h=self.num_heads)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # norm over inputs
                updates = torch.einsum('...is,...id->...sd', attn,
                                       v[:, t])  # Shape: [batch_size, num_heads, num_slots, num_inp].
                updates = rearrange(updates, 'b h n_s d -> b n_s (h d)')
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.reshape(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)

                slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

        attns_collect = torch.stack(attns_collect, dim=1)  # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)  # B, T, num_slots, slot_size

        return slots_collect, attns_collect

if __name__ == "__main__":
    # test
    slot_attn = MultiHeadSTEVESA(
        num_iterations=3, 
        num_slots=24, 
        num_heads=1,
        input_size=192, # unet_encoder.config.out_channels
        out_size=192, # unet.config.cross_attention_dim
        slot_size=192, 
        mlp_hidden_size=192,
        input_resolution=64, # unet_encoder.config.latent_size
        learnable_slot_init=False
    )
    slot_attn.save_config('./configs/movi-e/slot_attn')
    inputs = torch.randn(2, 1, 192, 64, 64)
    slots_collect, attns_collect = slot_attn(inputs)
    print(slots_collect.shape)
    pass