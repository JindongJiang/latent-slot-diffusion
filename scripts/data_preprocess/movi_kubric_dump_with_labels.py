# conda create -n tfds && conda activate tfds && pip install tensorflow-datasets gcfs tqdm pillow
import os
from tqdm.auto import tqdm
import json
import argparse
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_split', 
    type=str, 
    default='movi-a',
    choices=['movi-a', 'movi-b', 'movi-c', 'movi-d', 'movi-e', 'movi-f'],
    help='Dataset split to use (movi-a, movi-b, movi-c, movi-d, movi-e, movi-f)'
)
parser.add_argument(
    '--data_dir',
    type=str,
    default='/path_to_your_movi/',
    help='Directory to save the dataset'
)

args = parser.parse_args()

ds, ds_info = tfds.load(f"{args.dataset_split.replace('-', '_')}/256x256:1.0.0", 
                        data_dir="gs://kubric-public/tfds", with_info=True)

for section in ["train", "validation", "test"]:
    out_path_images = os.path.join(args.data_dir, f'{args.dataset_split}/{args.dataset_split}-{section}-with-label/images')
    out_path_labels = os.path.join(args.data_dir, f'{args.dataset_split}/{args.dataset_split}-{section}-with-label/labels')

    try:
        dataset = tfds.as_numpy(ds[section])
        data_iter = iter(dataset)
    except:
        continue

    # to_tensor = transforms.ToTensor()

    class JsonEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tf.RaggedTensor):
                return obj.to_tensor().numpy().tolist()
            elif isinstance(obj, bytes):
                return obj.decode()

            return json.JSONEncoder.default(self, obj)

    b = 0
    progress_bar = tqdm(
        range(0, len(dataset)),
        initial=0,
        desc=f"{section} Steps",
        position=0, leave=True
    )
    for record in data_iter:
        video = record['video']
        segments = record['segmentations']
        instances = {}
        for k, v in record['instances'].items():
            if isinstance(v, tf.RaggedTensor):
                v = v.to_tensor().numpy()
            instances[k] = v
        T, *_ = video.shape

        # setup dirs
        path_vid_images = os.path.join(out_path_images, f"{b:08}")
        os.makedirs(path_vid_images, exist_ok=True)
        path_vid_labels = os.path.join(out_path_labels, f"{b:08}")
        os.makedirs(path_vid_labels, exist_ok=True)

        for t in range(T):
            img = video[t]
            seg = segments[t]
            Image.fromarray(img).save(os.path.join(path_vid_images, f"{t:08}_image.png"), optimize=False)
            Image.fromarray(seg[..., 0]).save(os.path.join(path_vid_labels, f"{t:08}_segment.png"), optimize=False)

            instances_t = {}
            for k, v in instances.items():
                if len(v.shape) > 1 and v.shape[1] == T:
                    instances_t[k] = v[:, t]
                else:
                    instances_t[k] = v
            with open(os.path.join(path_vid_labels, f"{t:08}_instances.json"), 'w') as f:
                json.dump(instances_t, f, cls=JsonEncoder)

        b += 1
        progress_bar.update(1)