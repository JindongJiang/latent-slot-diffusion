import copy
import random
import distinctipy
from PIL import Image

import torch
import numpy as np
from torch import nn

from torchvision import transforms
from einops import rearrange, repeat
import matplotlib.cm as mpl_color_map
from torchvision.utils import make_grid
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class SinCosPositionalEmbedding2D(nn.Module):

    def __init__(self, embed_dim, grid_size, learnable_gamma=True):
        super().__init__()
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed = rearrange(pos_embed, '(h w) d -> 1 d h w', h=grid_size, w=grid_size)
        # self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float(), requires_grad=False)
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(1.), requires_grad=True)
        else:
            self.gamma = 1.

        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float())

    def forward(self, input):
        """
        input: batch_size x d_model x H x W
        """
        return input + self.gamma * self.pos_embed


class LearnedPositionalEmbedding1D(nn.Module):

    def __init__(self, num_inputs, input_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, input):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        return self.dropout(input + self.pe)


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        # self.projection = conv2d(4, channels, 1)
        self.projection = nn.Conv2d(4, channels, 1)
        # self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

        self.register_buffer('pe', self.build_grid(image_size).unsqueeze(0))

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='xy')
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


def random_colors(N, randomize=False, rng=42):
    colors = distinctipy.get_colors(N, rng=rng)
    if randomize:
        random.shuffle(colors)
    return colors

class ColorMask(object):
    def __init__(self, num_slots, log_img_size, norm_mean, 
                 norm_std, rng=42, img_tmp_pth=None, reshape_first=False):
        self.img_tmp_pth = img_tmp_pth
        self.num_slots = num_slots
        self.log_img_size = log_img_size
        self.color = torch.tensor(random_colors(num_slots, randomize=False, rng=rng))
        self.log_image_resize = transforms.Resize(log_img_size,
                                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=True)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.img_unnorm = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=1 / torch.tensor(norm_std)),
            transforms.Normalize(mean=-torch.tensor(norm_mean),
                                 std=[1., 1., 1.])
        ])
        self.reshape_first = reshape_first

    def apply_colormap_on_image(self, org_im, activation, colormap_name, alpha=0.5):
        """
            Apply heatmap on image
        Args:
            org_img (PIL img): Original image
            activation_map (numpy arr): Activation map (grayscale) 0-255
            colormap_name (str): Name of the colormap
        """
        # Get colormap
        color_map = mpl_color_map.get_cmap(colormap_name)
        no_trans_heatmap = color_map(activation)
        # Change alpha channel in colormap to make sure original image is displayed
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = alpha
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

        # Apply heatmap on image
        heatmap_on_image = Image.new("RGBA", org_im.size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
        return no_trans_heatmap, heatmap_on_image

    def _apply_mask(self, image, mask, alpha=0.5, color=None):

        B, C, H, W = image.size()
        B, N, H, W = mask.size()

        image = image.clone()
        mask_only = torch.ones_like(image)

        if color is None:
            color = random_colors(N)

        for n in range(N):
            for c in range(3):
                image[..., c, :, :] = torch.where(
                    mask[:, n] == 1,
                    image[..., c, :, :] * (1 - alpha) + alpha * (color[n][c] if isinstance(color, list) else
                                                                 color[..., n, c][..., None, None]),
                    image[..., c, :, :]
                )
                mask_only[..., c, :, :] = torch.where(
                    mask[:, n] == 1,
                    mask_only[..., c, :, :] * (1 - 0.99) + 0.99 * (color[n][c] if isinstance(color, list) else
                                                                   color[..., n, c][..., None, None]),
                    mask_only[..., c, :, :]
                )
        return image, mask_only

    def get_heatmap(self, img, attn, recon=None, mask_pred_sorted=None, return_all=False):
        '''

                :param img: b, c, h, w
                :param attn: b, s, h, w
                :param name:
                :param global_step:
                :return:
                '''
        img = img.to(torch.device('cpu'), dtype=torch.float32)
        attn = attn.to(torch.device('cpu'), dtype=torch.float32)
        if recon is not None:
            if not isinstance(recon, list):
                recon = [recon]
            recon = [r.to(torch.device('cpu'), dtype=torch.float32) for r in recon]
        if mask_pred_sorted is not None:
            mask_pred_sorted = mask_pred_sorted.to(torch.device('cpu'))
        bs, inp_channel, h, w = img.size()

        img = self.img_unnorm(img).clamp(0., 1.)
        if recon is not None:
            recon = [self.img_unnorm(r).clamp(0., 1.) for r in recon]

        if h > self.log_img_size:
            img = self.log_image_resize(img)
            h, w = img.shape[-2:]

        num_s = attn.size(1)
        # --------------------------------------------------------------------------
        # reshape first to get nicer visualization
        if self.reshape_first and (attn.shape[-2] != img.shape[-2] or attn.shape[-1] != img.shape[-1]):
            attn = transforms.Resize(size=img.shape[-2:], interpolation=transforms.InterpolationMode.BILINEAR,
                                     antialias=True)(attn)
            
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # get color map
        if mask_pred_sorted is None:
            mask_pred = (attn.argmax(1, keepdim=True) == torch.arange(attn.size(1))[None, :, None, None]).float()
        else:
            mask_pred = mask_pred_sorted

        if mask_pred.shape[-2] != img.shape[-2] or mask_pred.shape[-1] != img.shape[-1]:
            mask_pred = transforms.Resize(size=img.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST)(
                mask_pred)

        # b c h w
        img_overlay, color_mask = self._apply_mask(img, mask_pred, alpha=0.5, color=self.color)

        # --------------------------------------------------------------------------

        if attn.shape[-2] != img.shape[-2] or attn.shape[-1] != img.shape[-1]:
            attn = transforms.Resize(size=img.shape[-2:], interpolation=transforms.InterpolationMode.BILINEAR,
                                     antialias=True)(attn)

        attn = rearrange(attn, 'b s h w -> b s (h w)')

        attn = rearrange(attn, 'b s h_w -> (b s) h_w').detach().numpy()

        img_reshape = repeat(img, 'b c h w -> c (b s) (h w) ', s=num_s)

        img_pil = transforms.ToPILImage()(img_reshape)

        no_trans_heatmap, heatmap_on_image = self.apply_colormap_on_image(img_pil, attn, 'gray')

        heatmap_on_image = transforms.ToTensor()(heatmap_on_image.convert('RGB'))

        heatmap_on_image = rearrange(heatmap_on_image, 'c (b s) (h w) -> b s c h w', b=bs, c=inp_channel, h=h, w=w)

        grid_image = torch.cat([img[:, None], img_overlay[:, None], heatmap_on_image], dim=1)
        if recon is not None:
            if not isinstance(recon, list):
                recon = [recon]
            recon = [self.log_image_resize(r) if r.shape[-2] != h else r for r in recon]
            grid_image = torch.cat([*[r[:, None] for r in recon if r is not None], grid_image], dim=1)
        grid_image = make_grid(rearrange(grid_image, 'b n c h w -> (b n) c h w'),
                               nrow=grid_image.size(1), padding=1,
                               pad_value=0.8)
        if return_all:
            return grid_image, img_overlay, color_mask, heatmap_on_image
        return grid_image

    def log_heatmap(self, img, attn, recon=None, mask_pred_sorted=None, path=None):
        assert path is not None or self.img_tmp_pth is not None, 'path is None and img_tmp_pth is None'

        grid_image = self.get_heatmap(img, attn, recon, mask_pred_sorted)
        # save_image(grid_image, self.img_tmp_pth)
        ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        img_path = path if path is not None else self.img_tmp_pth
        im.save(img_path, optimize=True, quality=95)
        return img_path