import pytest
import torch

from mirei.constract_llm.eval.sentence_model.metric import (
    compute_alignment,
    compute_alignment_sq_distances,
    compute_pairwise_sq_distances,
    compute_uniformity,
    compute_uniformity_from_sq_distances,
)


@pytest.mark.parametrize(
    'z1, z2, expected',
    [
        # Two identical vectors → alignment=0
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 0.0),
        # Orthogonal vectors → alignment=2
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), 2.0),
    ],
)
def test_compute_alignment_param(z1, z2, expected):
    assert abs(compute_alignment(z1, z2) - expected) < 1e-6


@pytest.mark.parametrize(
    'z, t, expected',
    [
        # Two identical vectors → distance=0, uniformity=log(exp(0))=0
        (torch.tensor([[1.0, 0.0], [1.0, 0.0]]), 2.0, 0.0),
        # Orthogonal vectors → distance=2, uniformity=log(exp(-2*2))= -4
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 2.0, -4.0),
    ],
)
def test_compute_uniformity_param(z, t, expected):
    assert abs(compute_uniformity(z, t=t) - expected) < 1e-6


@pytest.mark.parametrize(
    'z1, z2, expected',
    [
        # Identical vs orthogonal pairs → per-pair squared distances [0, 2]
        (
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0.0, 2.0]),
        ),
        # Normalization is applied before computing distances
        (torch.tensor([[2.0, 0.0]]), torch.tensor([[5.0, 0.0]]), torch.tensor([0.0])),
    ],
)
def test_compute_alignment_sq_distances(z1, z2, expected):
    sq_distances = compute_alignment_sq_distances(z1, z2)
    assert torch.allclose(sq_distances, expected, atol=1e-6)


def test_compute_alignment_matches_mean_of_sq_distances():
    z1 = torch.randn(8, 4)
    z2 = torch.randn(8, 4)
    expected = compute_alignment_sq_distances(z1, z2).mean().item()
    assert abs(compute_alignment(z1, z2) - expected) < 1e-6


def test_compute_pairwise_sq_distances():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sq_distances = compute_pairwise_sq_distances(z)
    # Two off-diagonal entries, both orthogonal → squared distance 2
    assert sq_distances.shape == (2,)
    assert torch.allclose(sq_distances, torch.tensor([2.0, 2.0]), atol=1e-6)


def test_compute_uniformity_matches_from_sq_distances():
    z = torch.randn(8, 4)
    expected = compute_uniformity_from_sq_distances(compute_pairwise_sq_distances(z), t=2.0)
    assert abs(compute_uniformity(z, t=2.0) - expected) < 1e-6
