import numpy as np
import pytest
import torch

from mirei.constract_llm.eval.analysis.metrics import (
    ablate_dims,
    attention_summary,
    effective_rank,
    linear_cka,
    mutual_knn_overlap,
    ordered_text_sha256,
    outlier_stats,
    position_contribution,
    spearman,
)


def test_attention_summary_causal_backward_mass_zero():
    num_heads, seq_len = 2, 6
    attn = torch.tril(torch.ones(seq_len, seq_len))
    attn = attn / attn.sum(dim=-1, keepdim=True)
    attn = attn.unsqueeze(0).expand(num_heads, -1, -1)
    mask = torch.ones(seq_len, dtype=torch.bool)
    result = attention_summary(attn, mask)
    assert len(result['per_head']['backward_mass']) == num_heads
    assert result['mean']['backward_mass'] == pytest.approx(0.0, abs=1e-6)


def test_attention_summary_uniform_entropy_one():
    num_heads, seq_len = 3, 5
    attn = torch.full((num_heads, seq_len, seq_len), 1.0 / seq_len)
    mask = torch.ones(seq_len, dtype=torch.bool)
    result = attention_summary(attn, mask)
    assert result['mean']['entropy'] == pytest.approx(1.0, abs=1e-6)
    for value in result['per_head']['entropy']:
        assert value == pytest.approx(1.0, abs=1e-6)


def test_attention_summary_sink_mass_one():
    num_heads, seq_len = 2, 4
    attn = torch.zeros(num_heads, seq_len, seq_len)
    attn[:, :, 0] = 1.0
    mask = torch.ones(seq_len, dtype=torch.bool)
    result = attention_summary(attn, mask)
    assert result['mean']['sink_mass'] == pytest.approx(1.0, abs=1e-6)
    # Only the first query row attends to itself (j == i == 0).
    assert result['mean']['diag_mass'] == pytest.approx(1.0 / seq_len, abs=1e-6)


def test_attention_summary_renormalizes_over_valid_keys():
    num_heads, seq_len, valid_len = 2, 8, 4
    attn = torch.full((num_heads, seq_len, seq_len), 1.0 / seq_len)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[:valid_len] = True
    result = attention_summary(attn, mask)
    assert result['mean']['entropy'] == pytest.approx(1.0, abs=1e-6)
    assert result['mean']['sink_mass'] == pytest.approx(1.0 / valid_len, abs=1e-6)


def test_position_contribution_identical_tokens():
    seq_len, hidden_dim, num_bins = 20, 8, 10
    token_states = torch.ones(seq_len, hidden_dim)
    mask = torch.ones(seq_len, dtype=torch.bool)
    result = position_contribution(token_states, mask, num_bins=num_bins)
    assert result['num_valid_tokens'] == seq_len
    assert sum(result['count']) == seq_len
    assert sum(result['norm_share']) == pytest.approx(1.0, abs=1e-6)
    for bin_id in range(num_bins):
        assert result['count'][bin_id] == seq_len // num_bins
        assert result['cos_mean'][bin_id] == pytest.approx(1.0, abs=1e-5)
        assert result['norm_share'][bin_id] == pytest.approx(1.0 / num_bins, abs=1e-5)


def test_position_contribution_ignores_masked_tokens():
    token_states = torch.ones(6, 4)
    token_states[4:] = 100.0
    mask = torch.tensor([True, True, True, True, False, False])
    result = position_contribution(token_states, mask, num_bins=2)
    assert result['num_valid_tokens'] == 4
    assert result['norm_share'][0] == pytest.approx(0.5, abs=1e-6)


def test_outlier_stats_detects_high_variance_dim():
    torch.manual_seed(0)
    embeddings = torch.randn(500, 16)
    embeddings[:, 3] *= 20.0
    result = outlier_stats(embeddings, top_k=4)
    assert result['top_variance_dims'][0] == 3
    assert result['top1_variance_share'] > 0.9
    assert result['topk_variance_share'] >= result['top1_variance_share']
    assert 0.0 < result['top1_dim_mean_sq_norm_contribution'] <= 1.0
    assert len(result['top_kurtosis_dims']) == 4


def test_ablate_dims_zeroes_only_selected_dims():
    torch.manual_seed(1)
    embeddings = torch.randn(4, 6)
    original = embeddings.clone()
    ablated = ablate_dims(embeddings, [1, 4])
    assert torch.all(ablated[:, 1] == 0)
    assert torch.all(ablated[:, 4] == 0)
    kept = [0, 2, 3, 5]
    assert torch.equal(ablated[:, kept], original[:, kept])
    # The input tensor is not modified in place.
    assert torch.equal(embeddings, original)


def test_effective_rank_isotropic_gaussian():
    torch.manual_seed(2)
    hidden_dim = 16
    embeddings = torch.randn(2000, hidden_dim)
    result = effective_rank(embeddings)
    assert result['rankme'] > 0.9 * hidden_dim
    assert result['participation_ratio'] > 0.8 * hidden_dim
    assert result['top10_singular_share'] < 1.0


def test_effective_rank_rank_one_data():
    torch.manual_seed(3)
    direction = torch.randn(16)
    coefficients = torch.randn(500, 1)
    embeddings = coefficients @ direction.unsqueeze(0)
    result = effective_rank(embeddings)
    assert result['rankme'] < 1.2
    assert result['top1_singular_share'] > 0.99


def test_linear_cka_invariant_to_orthogonal_rotation():
    torch.manual_seed(4)
    x = torch.randn(200, 12)
    q, _ = torch.linalg.qr(torch.randn(12, 12))
    y = x @ q
    assert linear_cka(x, y) == pytest.approx(1.0, abs=1e-4)


def test_linear_cka_independent_random_is_small():
    torch.manual_seed(5)
    x = torch.randn(500, 20)
    y = torch.randn(500, 20)
    assert linear_cka(x, y) < 0.2


def test_mutual_knn_overlap_identical_data():
    torch.manual_seed(6)
    x = torch.randn(50, 8)
    assert mutual_knn_overlap(x, x, k=5) == pytest.approx(1.0)


def test_spearman_monotonic_sequences():
    a = np.arange(1.0, 11.0)
    assert spearman(a, np.exp(a)) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_ordered_text_sha256_is_order_and_content_sensitive() -> None:
    base = ordered_text_sha256(['a', 'bc'])
    assert base == ordered_text_sha256(['a', 'bc'])
    assert base != ordered_text_sha256(['bc', 'a'])
    assert base != ordered_text_sha256(['a', 'bd'])
    assert base != ordered_text_sha256(['ab', 'c'])
