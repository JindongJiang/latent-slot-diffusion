import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
from torch import nn
from diffusers.models.unet_2d import UNet2DModel, UNet2DOutput
from typing import Any, Dict, List, Optional, Tuple, Union


class UNetEncoder(UNet2DModel):

    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        input_resolution: int = 256,
        input_channels: int = 3,
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
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            **kwargs
        )

        self.register_to_config(
            input_resolution=input_resolution,
            input_channels=input_channels,
        )
        downscale_stride = input_resolution // sample_size
        self.downscale_cnn = nn.Conv2d(input_channels, in_channels, kernel_size=downscale_stride, stride=downscale_stride)
        self.original_forward = super().forward

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> Union[UNet2DOutput, Tuple]:

        sample = self.downscale_cnn(sample)
        return self.original_forward(sample, timestep=0, class_labels=None).sample

if __name__ == "__main__":
    block_out_channels = (128, 128, 256, 512) # each of then should be divisible by the least common multiple of (80, 32, 15)

    unet_encoder = UNetEncoder(
        sample_size= 64,
        in_channels = 128, # input channel of the unet conv, not the input image
        out_channels = 192, 
        down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels = block_out_channels,
        layers_per_block = 1,
        attention_head_dim = 8,
        norm_num_groups = 16,
        input_resolution=256,
        input_channels=3,
    )

    # save the config only if pass this test
    image_cond = torch.randn(1, 3, 256, 256)
    out = unet_encoder(image_cond)

    unet_encoder.save_config("./configs/movi-e/backbone/")   
    pass