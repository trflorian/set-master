import albumentations as A
import numpy as np
import pytest
import torch

from set_master.training.augmentation import CardAugmentation


@pytest.mark.parametrize("horizontal,vertical", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_flips_preserve_semantic_short_edge_and_clockwise_winding(horizontal: int, vertical: int) -> None:
    augmentation = CardAugmentation()
    augmentation.transform = A.Compose(
        [
            A.HorizontalFlip(p=float(horizontal)),
            A.VerticalFlip(p=float(vertical)),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        seed=42,
    )
    corners = np.float32([[10, 10], [40, 10], [40, 80], [10, 80]])
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 41.0, 81.0]]),
        "labels": torch.ones(1, dtype=torch.int64),
        "keypoints": torch.tensor([[[10.0, 10.0, 2.0], [40.0, 10.0, 1.0], [40.0, 80.0, 2.0], [10.0, 80.0, 1.0]]]),
    }
    _, result = augmentation(np.zeros((100, 100, 3), dtype=np.uint8), target, [[corners]])
    points = result["keypoints"][0, :, :2].numpy()
    assert np.linalg.norm(points[1] - points[0]) == 30
    assert np.linalg.norm(points[2] - points[1]) == 70
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    assert np.all(edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0] > 0)
    original = corners.copy()
    if horizontal:
        original[:, 0] = 99 - original[:, 0]
    if vertical:
        original[:, 1] = 99 - original[:, 1]
    indices = np.linalg.norm(points[:, None] - original[None], axis=2).argmin(axis=1)
    np.testing.assert_array_equal(result["keypoints"][0, :, 2].numpy(), np.array([2.0, 1.0, 2.0, 1.0])[indices])
