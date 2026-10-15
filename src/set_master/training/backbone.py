from collections import OrderedDict
from pathlib import Path

import torch
from dinov3.hub.backbones import dinov3_convnext_tiny
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork


class FrozenDinoFPN(nn.Module):
    out_channels = 256

    def __init__(self, weights_path: str, initialize_pretrained: bool = True) -> None:
        super().__init__()
        if initialize_pretrained and not Path(weights_path).is_file():
            raise FileNotFoundError(f"DINOv3 pretrained weights not found: {weights_path}")
        self.encoder = dinov3_convnext_tiny(pretrained=False, patch_size=None)
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
