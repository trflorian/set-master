import pytest
import torch
from torchvision.models.detection.roi_heads import keypointrcnn_loss

from set_master.training.models.corners import cyclic_keypoint_loss, order_and_filter_corners


def test_corner_ordering_preserves_scores_and_filters_invalid_detections() -> None:
    points = torch.tensor([
        [[0, 0, 1], [10, 10, 1], [10, 0, 1], [0, 10, 1]],
        [[0, 0, 1], [10, 0, 1], [10, 0, 1], [0, 10, 1]],
        [[0, 0, 1], [10, 0, 1], [3, 3, 1], [0, 10, 1]],
        [[0, 0, 1], [10, 0, 1], [10, float('nan'), 1], [0, 10, 1]],
        [[0, 0, 1], [0.1, 0, 1], [0.1, 0.1, 1], [0, 0.1, 1]],
    ])
    result = order_and_filter_corners({
        'keypoints': points, 'keypoints_scores': torch.arange(20).reshape(5, 4),
        'boxes': torch.zeros(5, 4), 'scores': torch.arange(5), 'labels': torch.ones(5),
    })
    torch.testing.assert_close(result['keypoints'][0, :, :2], torch.tensor([[0., 0.], [10., 0.], [10., 10.], [0., 10.]]))
    assert result['keypoints_scores'].tolist() == [[0, 2, 1, 3]]
    assert result['scores'].tolist() == [0]
    empty = order_and_filter_corners({name: value[:0] for name, value in result.items()})
    assert empty['keypoints'].shape == (0, 4, 3)


@pytest.mark.parametrize('shift', range(4))
def test_cyclic_loss_matches_best_assignment_per_roi(shift: int) -> None:
    boxes = [torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]])]
    points = torch.tensor([[[0., 0., 2.], [4., 0., 2.], [4., 4., 2.], [0., 4., 2.]]])
    targets = [points.expand(2, -1, -1).clone()]
    matched = [torch.tensor([0, 1])]
    logits = torch.full((2, 4, 4, 4), -5.)
    for channel, (y, x) in enumerate([(0, 0), (0, 3), (3, 3), (3, 0)]):
        logits[:, channel, y, x] = 5
    logits[1] = logits[1].roll(2, dims=0)
    logits.requires_grad_()
    baseline = keypointrcnn_loss(logits[:1], [boxes[0][:1]], [points], [torch.tensor([0])])
    targets[0] = targets[0].roll(shift, dims=1)
    loss = cyclic_keypoint_loss(logits, boxes, targets, matched)
    if shift % 2 == 0:
        torch.testing.assert_close(loss, baseline)
    else:
        assert loss > baseline + 5
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_cyclic_loss_handles_invisible_and_empty_targets() -> None:
    for count in (0, 1):
        logits = torch.randn(count, 4, 4, 4, requires_grad=True)
        loss = cyclic_keypoint_loss(
            logits, [torch.tensor([[0., 0., 4., 4.]]).expand(count, -1)],
            [torch.zeros(count, 4, 3)], [torch.arange(count)],
        )
        assert loss.item() == 0
        loss.backward()


def test_corner_ordering_preserves_predicted_starting_edge() -> None:
    points = torch.tensor([[[10., 0., 1.], [10., 10., 1.], [0., 10., 1.], [0., 0., 1.]]])
    result = order_and_filter_corners({
        'keypoints': points, 'keypoints_scores': torch.ones(1, 4),
        'boxes': torch.zeros(1, 4), 'scores': torch.ones(1), 'labels': torch.ones(1),
    })
    torch.testing.assert_close(result['keypoints'], points)
