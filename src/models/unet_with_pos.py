import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, \
    UNetMidBlock2DCrossAttn, UNetMidBlock2DSimpleCrossAttn
from src.models.utils import CartesianPositionalEmbedding, SinCosPositionalEmbedding2D, \
    LearnedPositionalEmbedding1D
    
def get_pos_emb(dim, resolution, pos_type):

    if pos_type == 'cartesian':
        return nn.Sequential(
                    Rearrange('b (h w) c -> b c h w', h=resolution),
                    CartesianPositionalEmbedding(dim, resolution),
                    Rearrange('b c h w -> b (h w) c')
                )
    elif pos_type == 'sincos':
        return nn.Sequential(
                    Rearrange('b (h w) c -> b c h w', h=resolution),
                    SinCosPositionalEmbedding2D(dim, resolution),
                    Rearrange('b c h w -> b (h w) c')
                )
    elif pos_type == 'learned':
        return LearnedPositionalEmbedding1D(resolution**2, dim)
    elif pos_type == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError

def hack_unet_for_pos(unet, pos_type='cartesian'):
    # hack the unet to add positional embedding to the transformer blocks
    blocks_to_add_pos = ["down_blocks", "up_blocks", "mid_block"]
    resolutions = {}
    resolutions["down_blocks"] = [unet.config.sample_size]
    for block in unet.down_blocks:
        if block.downsamplers is not None:
            resolutions["down_blocks"].append(resolutions["down_blocks"][-1] // 2)
        else:
            resolutions["down_blocks"].append(resolutions["down_blocks"][-1])
    resolutions["mid_block"] = [resolutions["down_blocks"][-1]]
    resolutions["up_blocks"] = [resolutions["down_blocks"][-1]]
    resolutions["down_blocks"] = resolutions["down_blocks"][:-1]
    for block in unet.up_blocks:
        if block.upsamplers is not None:
            resolutions["up_blocks"].append(int(resolutions["up_blocks"][-1] * 2))
        else:
            resolutions["up_blocks"].append(resolutions["up_blocks"][-1])
    resolutions["up_blocks"] = resolutions["up_blocks"][:-1]
        
    for block_name in blocks_to_add_pos:
        blocks = getattr(unet, block_name)
        resolution_blocks = resolutions[block_name]
        if block_name == "mid_block":
            blocks = [blocks]
        for block_idx, block in enumerate(blocks):
            if isinstance(block, CrossAttnDownBlock2D) or isinstance(block, CrossAttnUpBlock2D) or \
                isinstance(block, UNetMidBlock2DCrossAttn) or isinstance(block, UNetMidBlock2DSimpleCrossAttn):
                for attention_block in block.attentions:
                    for transformer_block in attention_block.transformer_blocks:
                        transformer_block.attn1.pos_emb = get_pos_emb(
                            dim=transformer_block.attn1.to_q.in_features,
                            resolution=resolution_blocks[block_idx],
                            pos_type=pos_type
                        )
                        transformer_block.attn1.original_forward = \
                            transformer_block.attn1.forward
                        def forward_with_pos(self, *inputs, **kwargs):
                            hidden_states = inputs[0]
                            hidden_states = self.pos_emb(hidden_states)
                            out = self.original_forward(hidden_states, *inputs[1:], **kwargs)
                            return out

                        bound_method = forward_with_pos.__get__(
                            transformer_block.attn1, 
                            transformer_block.attn1.__class__
                        )
                        setattr(transformer_block.attn1, 'forward', bound_method)
                        
                        transformer_block.attn2.pos_emb = get_pos_emb(
                            dim=transformer_block.attn2.to_q.in_features,
                            resolution=resolution_blocks[block_idx],
                            pos_type=pos_type
                        )
                        transformer_block.attn2.original_forward = \
                            transformer_block.attn2.forward
                        def forward_with_pos(self, *inputs, **kwargs):
                            hidden_states = inputs[0]
                            hidden_states = self.pos_emb(hidden_states)
                            out = self.original_forward(hidden_states, *inputs[1:], **kwargs)
                            return out

                        bound_method = forward_with_pos.__get__(
                            transformer_block.attn2, 
                            transformer_block.attn2.__class__
                        )
                        setattr(transformer_block.attn2, 'forward', bound_method)
    return

class UNet2DConditionModelWithPos(UNet2DConditionModel):

    # since it is inherited from UNet2DConditionModel, it supports options like 
    # enable_gradient_checkpointing and enable_xformers_memory_efficient_attention
    # for efficient training and inference
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        pos_type: str ='cartesian',
        **kwargs
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            **kwargs
        )
        self.register_to_config(pos_type=pos_type)
        hack_unet_for_pos(self, pos_type=pos_type)


if __name__ == "__main__":
    unet = UNet2DConditionModelWithPos(
        sample_size = 32,
        in_channels = 4,
        out_channels = 4,
        cross_attention_dim=192, # slot size
        down_block_types = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types = (
            "UpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D"
        ),
        block_out_channels = (192, 384, 768, 768),
        layers_per_block = 2,
        norm_num_groups = 24,
        transformer_layers_per_block = 1,
        attention_head_dim = 24,
        pos_type='cartesian'
    )

    # need to pass this small test
    feat = torch.randn(1, 4, 32, 32)
    encoder_hidden_states = torch.randn(1, 10, 192)
    unet(feat, timestep=0, encoder_hidden_states=encoder_hidden_states)

    unet.save_config("./configs/movi-e/unet/")

    pass