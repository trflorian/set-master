# from .data.custom_transforms import *

import os
import sys
import time

import torch

from inspyrenet.lib import *
from inspyrenet.utils.misc import *
from .data.dataloader import *
from .data.custom_transforms import *


def mask_image(opencv_image):
    # Load config
    opt = ed(yaml.load(open("configs/Plus_Ultra_LR.yaml"), yaml.FullLoader))

    model = eval(opt.Model.name)(**opt.Model)
    model.load_state_dict(
        torch.load(os.path.join(opt.Test.Checkpoint.checkpoint_dir, "latest.pth"), map_location=torch.device("cpu")),
        strict=True,
    )

    model.eval()

    model.cuda()

    print("Model setup")

    save_dir = "results"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # sample_list = eval('ImageLoader')('images/000004.jpg', opt.Test.Dataset.transforms)

    custom_transforms = get_transform(opt.Test.Dataset.transforms)

    print("Starting timing....")

    start_time = time.time()

    image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).convert("RGB")

    print("Image loaded")

    shape = image.size[::-1]
    sample = {"image": image, "name": "my_img", "shape": shape, "original": image}
    sample = custom_transforms(sample)

    print("Custom transform applied")

    sample["image"] = sample["image"].unsqueeze(0)
    if "image_resized" in sample.keys():
        sample["image_resized"] = sample["image_resized"].unsqueeze(0)

    print("unsqueezed")
    # sample_list = eval('ImageLoader')(source, )

    # sample = custom_transforms(image)

    sample = to_cuda(sample)
    print("to cuda")

    with torch.no_grad():
        out = model(sample)

    print("Forwarded")

    pred = to_numpy(out["pred"], sample["shape"])

    img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)

    img = img.astype(np.uint8)

    mask = Image.fromarray(img)

    # save_paths = os.path.split(sample['name'])
    # if len(save_paths) > 1:
    #     parent_dir = os.path.join(*save_paths[:-1])
    #     os.makedirs(os.path.join(save_dir, parent_dir), exist_ok=True)
    #
    # mask.save(os.path.join(save_dir, sample['name'] + '.png'))

    end_time = time.time()

    print(f"Inference took {end_time - start_time:.2f}s")

    return mask
