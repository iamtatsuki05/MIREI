import pytest
import torch

from nlp.constract_llm.eval.sentence_model.metric import compute_alignment, compute_uniformity


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
