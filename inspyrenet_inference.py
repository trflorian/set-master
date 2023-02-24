import os
import cv2
import sys
import tqdm
import torch
import argparse

import numpy as np

from PIL import Image

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.misc import *
from data.dataloader import *
from data.custom_transforms import *


def mask_image(fn):
    # Load config
    opt = ed(yaml.load(open('configs/Plus_Ultra_LR.yaml'), yaml.FullLoader))

    source = fn#'../images_500x500/PXL_20230205_150355105.jpg'

    model = eval(opt.Model.name)(**opt.Model)
    model.load_state_dict(torch.load(os.path.join(
        opt.Test.Checkpoint.checkpoint_dir, 'latest.pth'), map_location=torch.device('cpu')), strict=True)

    model.eval()

    save_dir = 'results'

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    sample_list = eval('ImageLoader')(source, opt.Test.Dataset.transforms)

    samples = sample_list

    for sample in samples:
        with torch.no_grad():
            out = model(sample)

        pred = to_numpy(out['pred'], sample['shape'])

        img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)

        img = img.astype(np.uint8)

        mask = Image.fromarray(img)

        save_paths = os.path.split(sample['name'])
        if len(save_paths) > 1:
            parent_dir = os.path.join(*save_paths[:-1])
            os.makedirs(os.path.join(save_dir, parent_dir), exist_ok=True)

        mask.save(os.path.join(save_dir, sample['name'] + '.png'))
        return mask
