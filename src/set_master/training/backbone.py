from collections import OrderedDict
import importlib
from pathlib import Path
import sys

import torch
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork


class FrozenDinoFPN(nn.Module):
    out_channels = 256

    def __init__(self, weights_path: str, repo_path: str, initialize_pretrained: bool = True) -> None:
        super().__init__()
        if not Path(repo_path, "hubconf.py").is_file():
            raise FileNotFoundError(f"DINOv3 source checkout not found: {repo_path}")
        if initialize_pretrained and not Path(weights_path).is_file():
            raise FileNotFoundError(f"DINOv3 pretrained weights not found: {weights_path}")
        # Import just the official backbone entry point: hubconf also imports optional
        # segmentation/depth packages that this detector does not use.
        sys.path.insert(0, repo_path)
        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
            self.encoder = backbones.dinov3_convnext_tiny(pretrained=False, patch_size=None)
        finally:
            sys.path.remove(repo_path)
        if initialize_pretrained:
            self.encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], self.out_channels)

    def train(self, mode: bool = True) -> "FrozenDinoFPN":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(images, n=4, reshape=True, norm=True)
        return self.fpn(OrderedDict((str(i), feature) for i, feature in enumerate(features)))


