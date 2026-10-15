from collections import OrderedDict
import json
from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import cv2
import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from set_master.training.augmentation import CardAugmentation
from set_master.training.config import DetectorVariant
from set_master.training.data import CocoCardDataset
from set_master.training.evaluation import evaluate_detector
from set_master.training.models.box import CardDetector
from set_master.training.models.factory import create_detector
from set_master.training.config import DetectorConfig


class TinyFPN(nn.Module):
    out_channels = 8

    def __init__(self, *args: object) -> None:
        super().__init__()
        self.project = nn.Conv2d(3, 8, 1)

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        features = self.project(images)
        return OrderedDict((str(index), F.avg_pool2d(features, 2 ** (index + 1))) for index in range(4))


def card_target(height: int, width: int) -> dict[str, Tensor]:
    masks = torch.zeros((1, height, width), dtype=torch.uint8)
    masks[:, 8:24, 4:20] = 1
    return {
        "boxes": torch.tensor([[4.0, 8.0, 20.0, 24.0]]),
        "labels": torch.ones(1, dtype=torch.int64),
        "masks": masks,
        "iscrowd": torch.zeros(1, dtype=torch.int64),
        "image_id": torch.tensor(1),
    }


@pytest.mark.parametrize(
    "variant", [DetectorVariant.SEMANTIC, DetectorVariant.SEMANTIC_REFINED, DetectorVariant.INSTANCE]
)
def test_segmentation_trains_and_restores_original_mask_sizes(
    monkeypatch: pytest.MonkeyPatch, variant: DetectorVariant
) -> None:
    monkeypatch.setattr("set_master.training.models.box.FrozenDinoFPN", TinyFPN)
    model = create_detector(DetectorConfig(variant=variant, min_size=32, max_size=64), initialize_pretrained=False)
    if variant == DetectorVariant.INSTANCE:
        model.detector.rpn._pre_nms_top_n = {"training": 20, "testing": 20}
        model.detector.rpn._post_nms_top_n = {"training": 10, "testing": 10}
        model.detector.roi_heads.detections_per_img = 3
    images = [torch.rand(3, 32, 48), torch.rand(3, 48, 32)]
    targets = [card_target(32, 48), card_target(48, 32)]
    # A background-only scene must also have a finite training signal.
    targets[1] = {name: value[:0] if value.ndim else value for name, value in targets[1].items()}
    losses = model.detector(images, targets)
    loss = torch.stack(list(losses.values())).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        predictions = model(images)
        model.validation_step((images, targets), 0)
    for image, prediction in zip(images, predictions):
        masks = prediction["semantic_mask"] if variant.is_semantic else prediction["masks"]
        assert masks.shape[-2:] == image.shape[-2:]
        assert torch.isfinite(masks).all()
        assert ((masks >= 0) & (masks <= 1)).all()


def test_dataset_masks_and_augmentation_keep_instance_identity(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "scene.png"), np.zeros((32, 48, 3), dtype=np.uint8))
    annotations = [
        {
            "id": i + 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, 8, 16, 16],
            "segmentation": [[x, 8, x + 15, 8, x + 15, 23, x, 23]],
        }
        for i, x in enumerate([4, 20])
    ]
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "scene.png", "width": 48, "height": 32}],
                "categories": [{"id": 1, "name": "card"}],
                "annotations": annotations,
            }
        )
    )
    augmentation = CardAugmentation()
    augmentation.transform = A.Compose(
        [A.HorizontalFlip(p=1)], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )
    original = CocoCardDataset(tmp_path, DetectorVariant.INSTANCE)[0][1]
    flipped = CocoCardDataset(tmp_path, DetectorVariant.INSTANCE, augmentation)[0][1]
    torch.testing.assert_close(flipped["masks"], original["masks"].flip(-1))
    assert flipped["masks"].shape == (2, 32, 48)
    assert not (flipped["masks"][0].bool() & flipped["masks"][1].bool()).any()


class PerfectSemantic(CardDetector):
    variant = DetectorVariant.SEMANTIC

    def __init__(self) -> None:
        pl.LightningModule.__init__(self)
        self.register_parameter("anchor", nn.Parameter(torch.zeros(())))

    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        return [{"semantic_mask": card_target(*image.shape[-2:])["masks"].float()} for image in images]


class PerfectInstance(PerfectSemantic):
    variant = DetectorVariant.INSTANCE

    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        predictions = []
        for image in images:
            target = card_target(*image.shape[-2:])
            predictions.append(
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                    "scores": torch.ones(1),
                    "masks": target["masks"][:, None].float(),
                }
            )
        return predictions


@pytest.mark.parametrize(
    "variant", [DetectorVariant.SEMANTIC, DetectorVariant.SEMANTIC_REFINED, DetectorVariant.INSTANCE]
)
def test_segmentation_evaluation_exports_masks_and_correct_metrics(tmp_path: Path, variant: DetectorVariant) -> None:
    cv2.imwrite(str(tmp_path / "scene.png"), np.zeros((32, 48, 3), dtype=np.uint8))
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "scene.png"}],
                "categories": [{"id": 1, "name": "card"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [4, 8, 16, 16],
                        "segmentation": [[4, 8, 19, 8, 19, 23, 4, 23]],
                    }
                ],
            }
        )
    )
    model = PerfectSemantic() if variant.is_semantic else PerfectInstance()
    model.variant = variant
    result = evaluate_detector(
        model.eval(), CocoCardDataset(tmp_path, variant), tmp_path / "results", overlay_image_ids=(1,), progress=False
    )
    if variant.is_semantic:
        assert result.metrics["foreground_iou"] == 1
        assert result.metrics["dice"] == 1
        assert "map" not in result.metrics and "predictions" not in result.images[0]
        mask_path = "masks/000001.png"
    else:
        assert result.metrics["mask_map"] == pytest.approx(1)
        assert result.metrics["map"] == pytest.approx(1)
        mask_path = "masks/000001/card_000.png"
    assert cv2.imread(str(tmp_path / "results" / mask_path)).shape[:2] == (32, 48)
    assert (tmp_path / "results/overlay_000001.jpg").is_file()


def test_pyramid_loss_excludes_padding_and_detaches_coarse_consistency() -> None:
    from set_master.training.models.semantic_refined import pyramid_mask_losses

    truth = torch.zeros(1, 1, 32, 32)
    truth[:, :, :20, :19] = 1
    valid = torch.zeros_like(truth)
    valid[:, :, :20, :19] = 1
    logits = [torch.randn(1, 1, 8, 8, requires_grad=True), torch.randn(1, 1, 32, 32, requires_grad=True)]
    altered = []
    for logit in logits:
        coverage = F.interpolate(valid, size=logit.shape[-2:], mode="area")
        altered.append(torch.where(coverage > 0, logit, logit + 100))
    losses = pyramid_mask_losses(logits, truth, valid)
    changed = pyramid_mask_losses(altered, truth + (1 - valid), valid)
    for name in losses:
        torch.testing.assert_close(losses[name], changed[name])
    losses["loss_pyramid_consistency"].backward()
    assert logits[0].grad is None
    assert logits[1].grad is not None and logits[1].grad.abs().sum() > 0
    assert not logits[1].grad[valid == 0].any()


def test_refined_checkpoint_restores_same_predictions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from set_master.training.models.factory import load_detector

    monkeypatch.setattr("set_master.training.models.box.FrozenDinoFPN", TinyFPN)
    model = create_detector(
        DetectorConfig(variant=DetectorVariant.SEMANTIC_REFINED, min_size=32, max_size=64), initialize_pretrained=False
    ).eval()
    image = torch.rand(3, 37, 49)
    checkpoint = tmp_path / "refined.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    restored = load_detector(checkpoint)
    assert restored.variant == DetectorVariant.SEMANTIC_REFINED
    assert restored.variant.selection_metric == "foreground_iou"
    with torch.no_grad():
        torch.testing.assert_close(model([image])[0]["semantic_mask"], restored([image])[0]["semantic_mask"])
