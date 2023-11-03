import os
import glob
import random
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class GlobDataset(Dataset):
    def __init__(self, root, img_size, img_glob='*.png', 
                 data_portion=(),  random_data_on_portion=True,
                vit_norm=False, random_flip=False, vit_input_resolution=448):
        super().__init__()
        if isinstance(root, str) or not hasattr(root, '__iter__'):
            root = [root]
            img_glob = [img_glob]
        if not all(hasattr(sublist, '__iter__') for sublist in data_portion) or data_portion == (): # if not iterable or empty
            data_portion = [data_portion]
        self.root = root
        self.img_size = img_size
        self.episodes = []
        self.vit_norm = vit_norm
        self.random_flip = random_flip

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
            transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if vit_norm:
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
        return len(self.episodes)

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.episodes[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.random_flip:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = self.transform(image)
        example["pixel_values"] = pixel_values
        if self.vit_norm:
            image_vit = self.transform_vit(image)
            example["pixel_values_vit"] = image_vit
        return example

if __name__ == "__main__":
    dataset = GlobDataset(
        root="/research/projects/object_centric/shared_datasets/movi/movi-e/movi-e-train-with-label/images/",
        img_size=256,
        img_glob="**/*.png",
        data_portion=(0.0, 0.9)
    )
    pass