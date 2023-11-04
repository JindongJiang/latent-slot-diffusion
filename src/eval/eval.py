from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import random
import torch
import argparse
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from accelerate import Accelerator
from einops import rearrange, reduce
from sklearn.metrics import r2_score

from packaging import version
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from diffusers.utils.import_utils import is_xformers_available
from src.eval.eval_utils import GlobVideoDatasetWithLabel, ari, get_mask_cosine_distance, \
    hungarian_algorithm, clevrtex_label_reading, movi_label_reading
from accelerate.utils import set_seed

from src.models.backbone import UNetEncoder
from src.models.slot_attn import MultiHeadSTEVESA

parser = argparse.ArgumentParser()


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
    "--dataset_root",
    type=str,
    default=None,
    help="Path to the dataset root.", # note that we should use val split for some of kubric dataset such as the movi-e, text split is for ood eval
    required=True,
)
parser.add_argument(
    "--dataset_glob",
    type=str,
    default="**/*.png",
    help="Glob pattern for the dataset.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=50,
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
)

parser.add_argument(
    "--data_portion",
    nargs=2, type=float, metavar=('x', 'y'),
    default=()
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
    "--resolution",
    type=int,
    default=256,
    help=(
        "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution"
    ),
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
    "--linear_prob_train_portion",
    type=float,
    default=0.83, 
    help=(
        "The portion of the training data (in the testing slots) to use for attribute prediction tasks."
    ),
)

logger = get_logger(__name__)

args = parser.parse_args()

set_seed(args.seed)
torch.backends.cudnn.deterministic = True

accelerator = Accelerator(mixed_precision=args.mixed_precision)
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

max_num_obj = None
max_num_obj_dict = {'movi-a': 10, 'movi-e': 23, 'movi-c': 10,
                    'clevr_with_masks': 10, 'clevrtex': 10}
if "movi-c-test" in args.dataset_root:
    max_num_obj = 24
else:
    for k, v in max_num_obj_dict.items():
        if k in args.dataset_root:
            max_num_obj = v
            break
assert max_num_obj is not None, 'max_num_obj is not set'
if 'movi-a' in args.dataset_root:
    classification_label_names = ['color_label',
                                  'material_label', 'shape_label', 'size_label']
    regresion_label_names = ['image_positions']
elif 'movi-e' in args.dataset_root or 'movi-c' in args.dataset_root:
    classification_label_names = ['category']
    regresion_label_names = ['image_positions', 'bboxes_3d', 'scale']
elif 'clevr_with_masks' in args.dataset_root:
    classification_label_names = ['color', 'material', 'shape', 'size']
    regresion_label_names = ['x', 'y', 'z', 'pixel_coords', 'rotation']
elif 'clevrtex' in args.dataset_root:
    classification_label_names = ['color', 'material', 'shape', 'size']
    regresion_label_names = ['pixel_coords', 'rotation']
else:
    raise NotImplementedError

label_reading = clevrtex_label_reading if 'clevrtex' in args.dataset_root else movi_label_reading

if args.data_portion:
    data_portion = args.data_portion
else:
    data_portion = ()

test_dataset = GlobVideoDatasetWithLabel(root=args.dataset_root, 
                                         max_num_obj=max_num_obj,
                                         img_size=args.resolution,
                                         img_glob=args.dataset_glob,
                                         data_portion=data_portion,
                                         keys_to_log=classification_label_names + regresion_label_names,
                                         label_reading=label_reading,
                                         )

test_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': False,
}

test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)

backbone = UNetEncoder.from_pretrained(
    args.ckpt_path, subfolder="UNetEncoder".lower())
backbone = backbone.to(device=accelerator.device, dtype=weight_dtype)
# backbone.register_to_config(**backbone.config)
slot_attn = MultiHeadSTEVESA.from_pretrained(
    args.ckpt_path, subfolder="MultiHeadSTEVESA".lower())
slot_attn = slot_attn.to(device=accelerator.device, dtype=weight_dtype)
# backbone.register_to_config(**load_model.config)

if args.enable_xformers_memory_efficient_attention:
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

total_data = 0
ari_list_no_bg = []
# ari_list_with_bg = []
miou_list_no_bg_average_over_objects = []
mbo_list_no_bg_average_over_objects = []

miou_list_no_bg_average_over_images = []
mbo_list_no_bg_average_over_images = []

slots_list = []
# shape_label_list = []
label_continuous_dict = {
    k: [] for k in regresion_label_names
}

label_discrete_dict = {
    k: [] for k in classification_label_names
}

progress_bar = tqdm(
        range(0, len(test_loader)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        position=0, leave=True
    )

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        # image, true_mask, labels = batch_data
        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
        true_masks = batch["masks"].to(device=accelerator.device, dtype=weight_dtype)
        labels = {k: v.to(device=accelerator.device, dtype=weight_dtype) for k, v in batch["labels"].items()}

        feat = backbone(pixel_values)
        slots, attns = slot_attn.forward_slots(feat[:, None])  # for the time dimension
        slots = slots[:, 0]
        attns = attns[:, 0]

        attns = reduce(attns, 'b num_head (h w) num_s -> b num_s h w', h=int(np.sqrt(attns.shape[2])),
                       w=int(np.sqrt(attns.shape[2])), reduction='sum')
        attns = F.interpolate(attns, size=true_masks.shape[-1], mode='bilinear')
        attn_argmax = attns.argmax(dim=1, keepdim=False)
        attns_one_hot = \
            rearrange(F.one_hot(attn_argmax, num_classes=attns.shape[1]).float(
            ), 'b h w num_slots -> b num_slots h w')

        ari_list_no_bg.append(
            ari(true_masks.cpu(), attn_argmax.cpu(), num_ignored_objects=1))
        true_masks_one_hot = F.one_hot(
            true_masks.long(), num_classes=max_num_obj + 1).float()
        true_masks_one_hot = rearrange(
            true_masks_one_hot, 'b h w num_classes -> b num_classes h w')
 
        cost_matrix = get_mask_cosine_distance(
            true_masks_one_hot[..., None, :, :], attns_one_hot[..., None, :, :])  # attns or attns_one_hot

        if labels['visibility'].shape[1] >= max_num_obj + 1:
            selected_objects = labels['visibility']
            selected_objects[:, 0] = 0
            if len(selected_objects.shape) == 2:
                selected_objects = selected_objects[:, :, None]
            obj_idx_adjustment = 0  # multi_object_datasets
        else:
            selected_objects = \
                torch.cat([torch.zeros_like(labels['visibility'][:, 0:1]).to(labels['visibility'].device),
                           labels['visibility'] > 0], dim=1)[..., None]
            obj_idx_adjustment = 1  # movi series or clevrtex

        cost_matrix = cost_matrix * selected_objects + \
            10000 * (1 - selected_objects)
        _, indices = hungarian_algorithm(cost_matrix)

        for idx_in_batch, num_o in enumerate(labels['num_obj']):
            miou_no_bg_per_img = []
            mbo_no_bg_per_img = []
            for gt_idx, pred_idx in zip(indices[idx_in_batch][0], indices[idx_in_batch][1]):
                if selected_objects[idx_in_batch, ..., 0][
                        gt_idx] == 0:  # no gt_idx - 1 here because we added the background to the beginning
                    continue

                gt_map = (true_masks[idx_in_batch] == gt_idx)
                pred_map = (attn_argmax[idx_in_batch] == pred_idx)
                miou_list_no_bg_average_over_objects.append(
                    (gt_map & pred_map).sum() / (gt_map | pred_map).sum())
                miou_no_bg_per_img.append(
                    (gt_map & pred_map).sum() / (gt_map | pred_map).sum())

                iou_all = (gt_map[None] & attns_one_hot[idx_in_batch].bool()).sum((-2, -1)) / \
                          (gt_map[None] | attns_one_hot[idx_in_batch].bool()).sum(
                              (-2, -1))
                mbo_list_no_bg_average_over_objects.append(iou_all.max())
                mbo_no_bg_per_img.append(iou_all.max())

                # slot_obj = slots[idx_in_batch, 0, pred_idx]
                slot_obj = slots[idx_in_batch, pred_idx]
                slots_list.append(slot_obj)
                for k in label_continuous_dict.keys():
                    label_continuous_dict[k].append(
                        labels[k][idx_in_batch, gt_idx - obj_idx_adjustment])
                for k in label_discrete_dict.keys():
                    label_discrete_dict[k].append(
                        (labels[k][idx_in_batch, gt_idx - obj_idx_adjustment]).long())
            if miou_no_bg_per_img:
                miou_list_no_bg_average_over_images.append(
                    torch.stack(miou_no_bg_per_img).mean())
                mbo_list_no_bg_average_over_images.append(

        total_data += pixel_values.shape[0]
        progress_bar.update(1)


ari_list_no_bg = torch.cat(ari_list_no_bg)

print('ari_list_no_bg: ', ari_list_no_bg.mean())

miou_list_no_bg_average_over_objects = torch.mean(torch.stack(miou_list_no_bg_average_over_objects))

print('miou_list_no_bg_average_over_objects: ', miou_list_no_bg_average_over_objects.mean())

mbo_list_no_bg_average_over_objects = torch.mean(torch.stack(mbo_list_no_bg_average_over_objects))

print('mbo_list_no_bg_average_over_objects: ', mbo_list_no_bg_average_over_objects.mean())

# most of the methods actually use the following segmentation metrics average over images,
# however, we use the above version which compute the metrics over objects, empirically
# the below over-image version consistently gets a higher score

miou_list_no_bg_average_over_images = torch.mean(torch.stack(miou_list_no_bg_average_over_images))

print('miou_list_no_bg_average_over_images: ', miou_list_no_bg_average_over_images.mean())

mbo_list_no_bg_average_over_images = torch.mean(torch.stack(mbo_list_no_bg_average_over_images))

print('mbo_list_no_bg_average_over_images: ', mbo_list_no_bg_average_over_images.mean())

slots_list = torch.stack(slots_list, dim=0)

# for training use float32 only
weight_dtype = torch.float32
slots_list = slots_list.to(device=accelerator.device, dtype=weight_dtype)

label_continuous_dict = {
    k: torch.stack(v, dim=0).to(device=accelerator.device, dtype=weight_dtype) for k, v in label_continuous_dict.items()
}
# normalize each dimension to aproximately [-1, 1]
# we didn't do it in the paper but realized one 
# should use it to get better performance results
for k, v in label_continuous_dict.items():
    v_min = v.min(dim=0, keepdim=True)[0]
    v_max = v.max(dim=0, keepdim=True)[0]
    v = (v - v_min) / (v_max - v_min) * 2 - 1
    label_continuous_dict[k] = v
label_continuous_dict = {
    k: v[..., None] if len(v.shape) == 1 else v for k, v in label_continuous_dict.items()
}

label_discrete_dict = {
    k: torch.stack(v, dim=0).to(device=accelerator.device, dtype=torch.long) for k, v in label_discrete_dict.items()
}
label_discrete_dict = {
    k: v[..., 0] if len(v.shape) == 2 else v for k, v in label_discrete_dict.items()
}

slot_continuous_net_dict = {k: nn.Sequential(
    nn.Linear(slots_list.shape[1], slots_list.shape[1]),
    nn.ReLU(),
    nn.Linear(slots_list.shape[1], v.shape[1]),
).to(device=accelerator.device, dtype=weight_dtype) for k, v in label_continuous_dict.items()}
slot_discrete_net_dict = {k: nn.Linear(slots_list.shape[1], int(v.max() + 1)).to(device=accelerator.device, dtype=weight_dtype)
                          for k, v in label_discrete_dict.items()}

# split train and test data
train_portion = args.linear_prob_train_portion
# shuffle data and label in the same way
torch.random.manual_seed(args.seed)
shuffle_idx = torch.randperm(slots_list.shape[0])
for k, v in label_continuous_dict.items():
    label_continuous_dict[k] = label_continuous_dict[k][shuffle_idx]
for k, v in label_discrete_dict.items():
    label_discrete_dict[k] = label_discrete_dict[k][shuffle_idx]
slots_list = slots_list[shuffle_idx]
slots_list_train = slots_list[:int(slots_list.shape[0] * train_portion)]
slots_list_test = slots_list[int(slots_list.shape[0] * train_portion):]
label_continuous_dict_train = {
    k: v[:int(v.shape[0] * train_portion)] for k, v in label_continuous_dict.items()}
label_continuous_dict_test = {
    k: v[int(v.shape[0] * train_portion):] for k, v in label_continuous_dict.items()}
label_discrete_dict_train = {
    k: v[:int(v.shape[0] * train_portion)] for k, v in label_discrete_dict.items()}
label_discrete_dict_test = {
    k: v[int(v.shape[0] * train_portion):] for k, v in label_discrete_dict.items()}


params = []
for k, v in slot_continuous_net_dict.items():
    params = params + list(v.parameters())
for k, v in slot_discrete_net_dict.items():
    params = params + list(v.parameters())

optimizer = torch.optim.AdamW(params)
slot_continuous_loss = dict()
slot_discrete_loss = dict()

print('start property prediction training')

num_training_steps = 4000

progress_bar = tqdm(
    range(0, num_training_steps),
    initial=0,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
    position=0, leave=True
)

for epoch in range(num_training_steps):
    optimizer.zero_grad()
    for k, v in slot_continuous_net_dict.items():
        slot_continuous_loss[k] = F.mse_loss(
            v(slots_list_train), label_continuous_dict_train[k])
    for k, v in slot_discrete_net_dict.items():
        slot_discrete_loss[k] = F.cross_entropy(
            v(slots_list_train), label_discrete_dict_train[k])
    loss = sum(slot_continuous_loss.values()) + \
        sum(slot_discrete_loss.values())
    loss = loss / slots_list_train.shape[0]
    loss.backward()
    optimizer.step()
    progress_bar.update(1)

all_loss = dict()
for k, v in slot_continuous_net_dict.items():
    print(f'continuous {k}:', torch.mean(
        F.mse_loss(v(slots_list_test), label_continuous_dict_test[k])))
for k, v in slot_discrete_net_dict.items():
    print(f'discrete {k}:', torch.mean(
        (torch.argmax(v(slots_list_test), dim=1) == label_discrete_dict_test[k]).float()))
