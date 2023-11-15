from pathlib import Path


import os
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import random
import torch
import argparse
import importlib

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    # LDMTextToImagePipeline
    StableDiffusionPipeline,
)

import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
from torch import nn
from tqdm.auto import tqdm
# from tqdm.autonotebook import tqdm
from accelerate import Accelerator
from einops import rearrange, reduce
from sklearn.metrics import r2_score

from packaging import version
import torch.nn.functional as F
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate.logging import get_logger
from diffusers.utils.import_utils import is_xformers_available
from src.eval.eval_utils import GlobVideoDatasetWithLabel, ari, get_mask_cosine_distance, \
    hungarian_algorithm, clevrtex_label_reading, movi_label_reading
from accelerate.utils import ProjectConfiguration, set_seed

from src.models.utils import ColorMask
from src.models.backbone import UNetEncoder
from src.models.slot_attn import MultiHeadSTEVESA

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pretrained_model_name",
    type=str,
    default="stabilityai/stable-diffusion-2-1",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)

parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="Path to a checkpoint folder for the model.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=666,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=10,
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
)

parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)

parser.add_argument(
    "--validation_scheduler",
    type=str,
    default="DPMSolverMultistepScheduler",
    choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
    help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
)

parser.add_argument(
    "--vit_input_resolution",
    type=int,
    default=448,
    help=(
        "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution"
    ),
)

parser.add_argument(
    "--resolution",
    type=int,
    default=256,
    help=(
        "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution"
    ),
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./logs/images",
    help="Path to a output folder for logging images.",
)

parser.add_argument(
    "--cfg_list",
    nargs='+', type=str, metavar=('x', 'y'),
    default=(1.0, 1.1, 1.3, 1.5, 2.0, 3.0),
    help="List of classifier free guidance values.",
)

parser.add_argument(
    "--enable_xformers_memory_efficient_attention",
    action="store_true",
    help=(
        "Whether to use the memory efficient attention implementation of xFormers. This is an experimental feature"
        " and is only available for PyTorch >= 1.10.0 and xFormers >= 0.0.17."
    ),
)

parser.add_argument(
    "--path_to_coco",
    type=str,
    default="path_to_coco",
    help="Path to coco dataset.",
)

args = parser.parse_args()

class GlobDataset(Dataset):
    def __init__(self, coco_dir="/path_to_coco/", coco_split="val2017", 
                img_size=256, vit_input_resolution=448):
        super().__init__()

        self.coco_dir=coco_dir
        self.coco_split='val2017'
        annFile='{}/annotations/instances_{}.json'.format(self.coco_dir, self.coco_split)

        # Initialize the COCO api for instance annotations
        self.coco=COCO(annFile)

        img_ids = sorted(self.coco.getImgIds())
        self.imgs_info = self.coco.loadImgs(img_ids) # 5000

        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])

        self.transform_vit = transforms.Compose([
            transforms.Resize(vit_input_resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(vit_input_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    
    def __len__(self):
        return len(self.imgs_info)

    def __getitem__(self, i):
        img_path = os.path.join(self.coco_dir, 'images', self.coco_split, self.imgs_info[i]['file_name'])
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        example["pixel_values"] = self.transform(img)
        image_vit = self.transform_vit(img)
        example["pixel_values_vit"] = image_vit
        return example

test_dataset = GlobDataset(coco_dir=args.path_to_coco, vit_input_resolution=args.vit_input_resolution)

logger = get_logger(__name__)

set_seed(args.seed)
torch.backends.cudnn.deterministic = True

accelerator = Accelerator(mixed_precision=args.mixed_precision)
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

test_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': False,
}

test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)

if os.path.exists(os.path.join(args.ckpt_path, "UNetEncoder")):
    pretrain_backbone = False
    backbone = UNetEncoder.from_pretrained(
        args.ckpt_path, subfolder="UNetEncoder".lower())
    backbone = backbone.to(device=accelerator.device, dtype=weight_dtype)
else:
    pretrain_backbone = True
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    class DINOBackbone(torch.nn.Module):
        def __init__(self, dinov2):
            super().__init__()
            self.dinov2 = dinov2

        def forward(self, x):
            enc_out = self.dinov2.forward_features(x)
            return rearrange(
                enc_out["x_norm_patchtokens"], 
                "b (h w ) c -> b c h w",
                h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
            )
    backbone = DINOBackbone(dinov2).to(device=accelerator.device, dtype=weight_dtype)


pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name
    )

# todo: this is ugly solution
# use a more efficient scheduler at test time
module = importlib.import_module("diffusers")
scheduler_class = getattr(module, args.validation_scheduler)
scheduler = scheduler_class.from_config(pipeline.scheduler.config)

pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name, scheduler=scheduler
    )

pipeline = pipeline.to(accelerator.device)
pipeline.set_progress_bar_config(disable=True)

generator = None if args.seed is None else torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

slot_attn = MultiHeadSTEVESA.from_pretrained(
    args.ckpt_path, subfolder="MultiHeadSTEVESA".lower())
slot_attn = slot_attn.to(device=accelerator.device, dtype=weight_dtype)

colorizer = ColorMask(
        num_slots=slot_attn.config.num_slots,
        log_img_size=256,
        norm_mean=0,
        norm_std=1,
        reshape_first=True, # reshape first since the resolution is low
    )

if args.enable_xformers_memory_efficient_attention and not pretrain_backbone:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        backbone.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly")

progress_bar = tqdm(
        range(0, len(test_loader)),
        initial=0,
        desc="Steps",
        position=0, leave=True
    )

os.makedirs(args.output_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

        if pretrain_backbone:
            pixel_values_vit = batch["pixel_values_vit"].to(device=accelerator.device, dtype=weight_dtype)
            feat = backbone(pixel_values_vit)
        else:
            feat = backbone(pixel_values)
        slots, attns = slot_attn(feat[:, None])  # for the time dimension
        slots = slots[:, 0]
        attns = attns[:, 0]
        for seed in range(10):
            images = []
            for cfg in args.cfg_list:
                generator = torch.Generator(
                    device=accelerator.device).manual_seed(seed)
                images_gen = pipeline(
                        # .to(dtype=torch.float32) # needed?
                        prompt_embeds=slots,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=100,
                        generator=generator,
                        guidance_scale=cfg,
                        output_type="pt",
                    ).images
                images.append(images_gen)
            images = torch.stack(images, dim=1) # bs, cfg, 3, 256, 256
            grid_image = make_grid(rearrange(images, 'b n c h w -> (b n) c h w'),
                                nrow=images.size(1), padding=1,
                                pad_value=0.8)
            grid_image = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(grid_image)
            im.save(os.path.join(args.output_dir, f'{batch_idx}_images_gen_seed_{seed}.png'))

        grid_image = colorizer.get_heatmap(img=(pixel_values * 0.5 + 0.5),
                                           attn=reduce(
                                               attns, 'b num_h (h w) s -> b s h w', h=int(np.sqrt(attns.shape[-2])), 
                                               reduction='mean'
                                           ),
                                           recon=[])
        ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        print(f'Saving images to {args.output_dir}...')
        im.save(os.path.join(args.output_dir, f'{batch_idx}_segmentation.png'))
        if batch_idx > 10:
            break
