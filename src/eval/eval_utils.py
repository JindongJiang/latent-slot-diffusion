import os
import json
import glob
import torch
import random
import itertools
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from torch import Tensor, LongTensor
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple, Dict, Any, List

clevrtex_labels_to_idx = {
    'shape': {'cube': 0, 'cylinder': 1, 'monkey': 2, 'sphere': 3},
    'size': {'large': 0, 'medium': 1, 'small': 2},
    'color': {'blue': 0, 'brown': 1, 'cyan': 2, 'gray': 3, 'green': 4, 'purple': 5, 'red': 6, 'yellow': 7},
    'material': {
        'poliigonbricks01': 0,
        'poliigonbricksflemishred001': 1,
        'poliigonbrickspaintedwhite001': 2,
        'poliigoncarpettwistnatural001': 3,
        'poliigonchainmailcopperroundedthin001': 4,
        'poliigoncitystreetasphaltgenericcracked002': 5,
        'poliigoncitystreetroadasphalttwolaneworn001': 6,
        'poliigoncliffjagged004': 7,
        'poliigoncobblestonearches002': 8,
        'poliigonconcretewall001': 9,
        'poliigonfabricdenim003': 10,
        'poliigonfabricfleece001': 11,
        'poliigonfabricleatherbuffalorustic001': 12,
        'poliigonfabricrope001': 13,
        'poliigonfabricupholsterybrightanglepattern001': 14,
        'poliigongroundclay002': 15,
        'poliigongrounddirtforest014': 16,
        'poliigongrounddirtrocky002': 17,
        'poliigongroundforest003': 18,
        'poliigongroundforest008': 19,
        'poliigongroundforestmulch001': 20,
        'poliigongroundforestroots001': 21,
        'poliigongroundmoss001': 22,
        'poliigongroundsnowpitted003': 23,
        'poliigongroundtiretracks001': 24,
        'poliigoninteriordesignrugstarrynight001': 25,
        'poliigonmarble062': 26, 'poliigonmarble13': 27,
        'poliigonmetalcorrodedheavy001': 28,
        'poliigonmetalcorrugatedironsheet002': 29,
        'poliigonmetaldesignerweavesteel002': 30,
        'poliigonmetalpanelrectangular001': 31,
        'poliigonmetalspottydiscoloration001': 32,
        'poliigonmetalstainlesssteelbrushed': 33,
        'poliigonplaster07': 34, 'poliigonplaster17': 35,
        'poliigonroadcityworn001': 36,
        'poliigonrooftilesterracotta004': 37,
        'poliigonrustmixedonpaint012': 38,
        'poliigonrustplain007': 39,
        'poliigonsolarpanelspolycrystallinetypebframedclean001': 40,
        'poliigonstonebricksbeige015': 41,
        'poliigonstonemarblecalacatta004': 42,
        'poliigonterrazzovenetianmattewhite001': 43,
        'poliigontiles05': 44,
        'poliigontilesmarblechevroncreamgrey001': 45,
        'poliigontilesmarblesagegreenbrickbondhoned001': 46,
        'poliigontilesonyxopaloblack001': 47,
        'poliigontilesrectangularmirrorgray001': 48,
        'poliigonwallmedieval003': 49,
        'poliigonwaterdropletsmixedbubbled001': 50,
        'poliigonwoodfinedark004': 51,
        'poliigonwoodflooring044': 52,
        'poliigonwoodflooring061': 53,
        'poliigonwoodflooringmahoganyafricansanded001': 54,
        'poliigonwoodflooringmerbaubrickbondnatural001': 55,
        'poliigonwoodplanks028': 56,
        'poliigonwoodplanksworn33': 57,
        'poliigonwoodquarteredchiffon001': 58,
        'whitemarble': 59
    }
}


def cosine_similarity(a: Tensor, b: Tensor, eps: float = 1e-6):
    """Computes the cosine similarity between two tensors.
    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.
    Returns:
        The (batched) cosine similarity between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    dot_products = torch.matmul(a, b.transpose(1, 2))
    norm_a = (a * a).sum(dim=2).sqrt().unsqueeze(2)
    norm_b = (b * b).sum(dim=2).sqrt().unsqueeze(1)
    return dot_products / (torch.matmul(norm_a, norm_b) + eps)


def cosine_distance(a: Tensor, b: Tensor, eps: float = 1e-6):
    """Computes the cosine distance between two tensors, as 1 - cosine_similarity.
    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.
    Returns:
        The (batched) cosine distance between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    return 1 - cosine_similarity(a, b, eps)


def get_mask_cosine_distance(true_mask: Tensor, pred_mask: Tensor):
    """Computes the cosine distance between the true and predicted masks.
    Args:
        true_mask (Tensor): Tensor of shape (batch size, num objects, 1, H, W).
        pred_mask (Tensor): Tensor of shape (batch size, num slots, 1, H, W).
    Returns:
        The (batched) cosine similarity between the true and predicted masks, with
        shape (batch size, num objects, num slots).
    """
    return cosine_distance(true_mask.flatten(2).detach(), pred_mask.flatten(2).detach())


def hungarian_algorithm(cost_matrix: Tensor) -> Tuple[Tensor, LongTensor]:
    """Batch-applies the hungarian algorithm to find a matching that minimizes the overall cost.
    Returns the matching indices as a LongTensor with shape (batch size, 2, min(num objects, num slots)).
    The first column is the row indices (the indices of the true objects) while the second
    column is the column indices (the indices of the slots). The row indices are always
    in ascending order, while the column indices are not necessarily.
    The outputs are on the same device as `cost_matrix` but gradients are detached.
    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the row
    indices will be [1,2,3] and the column indices will be [0,2,1].
    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots)) with the
              costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects, num slots))
              containing the indices for the resulting matching.
    """

    # List of tuples of size 2 containing flat arrays
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.LongTensor(np.array(indices))
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    device = cost_matrix.device
    return smallest_cost_matrix.to(device), indices.to(device)


def ari(
        true_mask, pred_mask, num_ignored_objects: int
) -> torch.FloatTensor:
    """Computes the ARI score.
    Args:
        true_mask: tensor of shape [batch_size x *] where values go from 0 to the number of objects.
        pred_mask:  tensor of shape [batch_size x *] where values go from 0 to the number of objects.
        num_ignored_objects: number of objects (in ground-truth mask) to be ignored when computing ARI.
    Returns:
        a vector of ARI scores, of shape [batch_size, ].
    """
    true_mask = true_mask.flatten(1)
    pred_mask = pred_mask.flatten(1)
    not_bg = true_mask >= num_ignored_objects
    result = []
    batch_size = len(true_mask)
    for i in range(batch_size):
        ari_value = adjusted_rand_score(
            true_mask[i][not_bg[i]], pred_mask[i][not_bg[i]]
        )
        result.append(ari_value)
    result = torch.FloatTensor(result)  # shape (batch_size, )
    return result


def movi_label_reading(image_path, max_num_obj, keys_to_log=None):
    p_mask = str(image_path).replace('/images/', '/labels/').replace('_image.png', '_segment.png')
    p_labels = str(image_path).replace('/images/', '/labels/').replace('_image.png', '_instances.json')
    segment = np.array(Image.open(p_mask).convert('L'))
    with open(p_labels, 'rb') as f:
        label_dict = json.load(f)

    num_obj = len(label_dict['visibility'])
    labels = {'num_obj': num_obj}
    for k, v in label_dict.items():
        if np.array(v).ndim >= 3:
            # flatten the last two dimensions
            v = np.array(v)
            v = v.reshape(v.shape[0], -1).tolist()
        if k == 'visibility':
            continue
        if keys_to_log is None:
            labels[k] = torch.tensor(
                [i for i in v] + [v[-1] for _ in range(max_num_obj - num_obj)]
            )
        else:
            if k in keys_to_log:
                labels[k] = torch.tensor(
                    [i for i in v] + [v[-1] for _ in range(max_num_obj - num_obj)]
                )
    labels['visibility'] = torch.tensor(
        [i for i in label_dict['visibility']] +
        [0 for _ in range(max_num_obj - num_obj)]
    )
    if 'pixel_coords' in labels:
        labels['pixel_coords'] = labels['pixel_coords'][..., :2]
        labels['pixel_coords'][..., 0] -= 160
        labels['pixel_coords'][..., 0] /= 160

        labels['pixel_coords'][..., 1] -= 120
        labels['pixel_coords'][..., 1] /= 120

    return segment, labels

def clevrtex_label_reading(image_path, max_num_obj, keys_to_log=None):
    seg_path = image_path.replace('.png', '_flat.png')
    json_path = image_path.replace('.png', '.json')
    segment = Image.open(seg_path)
    w, h = segment.size
    min_side_len = min(w, h)
    left = (w - min_side_len) // 2
    top = (h - min_side_len) // 2
    right = (w + min_side_len) // 2
    bottom = (h + min_side_len) // 2
    segment = segment.crop((left, top, right, bottom))
    segment = np.array(segment).astype(np.uint8)
    with open(json_path, 'rb') as f:
        label_dict = json.load(f)
    objects_list = label_dict['objects']
    num_obj = len(objects_list)
    labels = {'num_obj': num_obj}
    for k in keys_to_log:
        labels[k] = []
    labels['visibility'] = []

    for obj_dict in objects_list:
        idx = obj_dict['index']
        labels['visibility'].append(np.any(segment == idx))
        for k in keys_to_log:
            v = obj_dict[k]
            if np.array(v).ndim >= 2:
                # flatten the last two dimensions
                v = np.array(v)
                v = v.reshape(v.shape[0], -1).tolist()
            if k in clevrtex_labels_to_idx:
                v = clevrtex_labels_to_idx[k][v]
            labels[k].append(v)
    for k, v in labels.items():
        if isinstance(v, int):
            continue
        if k == 'visibility':
            labels[k] = torch.tensor(
                [i for i in v] + [0 for _ in range(max_num_obj - num_obj)]
            ).float()
        else:
            labels[k] = torch.tensor([i for i in v] + [v[-1] for _ in range(max_num_obj - num_obj)]).float()

    if 'pixel_coords' in labels:
        labels['pixel_coords'] = labels['pixel_coords'][..., :2]
        labels['pixel_coords'][..., 0] -= w // 2
        labels['pixel_coords'][..., 0] /= w // 2

        labels['pixel_coords'][..., 1] -= h // 2
        labels['pixel_coords'][..., 1] /= h // 2

    return segment, labels

class GlobVideoDatasetWithLabel(Dataset):
    def __init__(self, root, img_size, max_num_obj, random_data_on_portion=True, 
                 img_glob='*.png', data_portion=(), keys_to_log=None, 
                 label_reading=movi_label_reading, return_path=False
                 ):
        
        super().__init__()
        if isinstance(root, str) or not hasattr(root, '__iter__'):
            root = [root]
            img_glob = [img_glob]
        if not all(hasattr(sublist, '__iter__') for sublist in data_portion) or data_portion == (): # if not iterable or empty
            data_portion = [data_portion]
        self.root = root
        self.img_size = img_size
        self.episodes = []

        for n, (r, g) in enumerate(zip(root, img_glob)):
            episodes = glob.glob(os.path.join(r, g), recursive=True)

            episodes = sorted(episodes)

            data_p = data_portion[n]

            assert (len(data_p) == 0 or len(data_p) == 2)
            if len(data_p) == 2:
                assert max(data_p) <= 1.0 and min(data_p) >= 0.0

            if data_p and data_p != (0., 1.):
                if random_data_on_portion:
                    random.Random(42).shuffle(episodes) # fix results
                episodes = \
                    episodes[int(len(episodes)*data_p[0]):int(len(episodes)*data_p[1])]

            self.episodes += episodes
        
        # resize the shortest side to img_size and center crop
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.max_num_obj = max_num_obj
        self.keys_to_log = keys_to_log
        self.label_reading = label_reading
        self.return_path = return_path

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        img_loc = self.episodes[idx]
        if self.keys_to_log:
            masks, labels = self.label_reading(img_loc, self.max_num_obj, keys_to_log=self.keys_to_log)
        else:
            masks, labels = 0, 0
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        example = {}
        example['pixel_values'] = tensor_image
        example['labels'] = labels
        example['masks'] = masks
        example['img_loc'] = img_loc
        return example