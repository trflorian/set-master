import pytest
import torch

from set_master.training.models.quadrilateral import (
    orient_quadrilaterals, portrait_axis_encoding, quadrilateral_geometry_loss,
)


@pytest.mark.parametrize('shift', range(4))
def test_geometry_does_not_require_a_card_relative_starting_corner(shift: int) -> None:
    points = torch.tensor([[[-.4, -.5], [.3, -.4], [.5, .5], [-.3, .4]]])
    prediction = points.roll(shift, dims=1).clone().requires_grad_()
    loss = quadrilateral_geometry_loss(prediction, points, torch.ones(1, 4, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_axis_encoding_accepts_half_turns_and_distinguishes_sideways() -> None:
    points = torch.tensor([[[0., 0.], [2., 0.], [2., 4.], [0., 4.]]])
    axis = portrait_axis_encoding(points)
    torch.testing.assert_close(axis, portrait_axis_encoding(points.roll(2, dims=1)))
    torch.testing.assert_close(axis, -portrait_axis_encoding(points.roll(1, dims=1)))
    # Encoding has no discontinuity when the directed axis reverses at 180 degrees.
    angle = torch.tensor([179., 181.]) * torch.pi / 180
    rotation = torch.stack((torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)), dim=1).reshape(2, 2, 2)
    rotated = torch.einsum('nij,nkj->nki', rotation, points.expand(2, -1, -1))
    assert torch.linalg.vector_norm(portrait_axis_encoding(rotated)[0] - portrait_axis_encoding(rotated)[1]) < .1


@pytest.mark.parametrize('shift', range(4))
def test_orientation_selects_short_edges_without_changing_geometry(shift: int) -> None:
    # Perspective can make the semantic short edges longer in the image.
    expected = torch.tensor([[[0., 0.], [8., 0.], [7., 3.], [1., 3.]]])
    prediction = expected.roll(shift, dims=1)
    result = orient_quadrilaterals(prediction, portrait_axis_encoding(expected))
    assert any(torch.allclose(result, expected.roll(offset, dims=1)) for offset in (0, 2))
    torch.testing.assert_close(result.sort(dim=1).values, prediction.sort(dim=1).values)
    sideways = orient_quadrilaterals(prediction, portrait_axis_encoding(expected.roll(1, dims=1)))
    assert any(torch.allclose(sideways, expected.roll(offset, dims=1)) for offset in (1, 3))


def test_geometry_handles_empty_and_invisible_targets() -> None:
    for count in (0, 1):
        prediction = torch.zeros(count, 4, 2, requires_grad=True)
        loss = quadrilateral_geometry_loss(prediction, torch.ones_like(prediction), torch.zeros(count, 4, dtype=torch.bool))
        assert loss.item() == 0
        loss.backward()
        result = orient_quadrilaterals(prediction.detach(), torch.zeros(count, 2))
        assert result.shape == (count, 4, 2)
