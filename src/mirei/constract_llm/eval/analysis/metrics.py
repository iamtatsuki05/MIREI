import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

_EPS = 1e-12


def attention_summary(attn: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    """
    Summarize one layer's attention map of a single sentence.

    Args:
        attn: torch.Tensor of shape (num_heads, seq_len, seq_len); rows are query positions.
        mask: torch.Tensor of shape (seq_len,); True for valid tokens.

    Returns:
        dict with 'per_head' ({metric: list[float] of length num_heads}) and
        'mean' ({metric: float averaged over heads}). Metrics: backward_mass, entropy,
        mean_distance, sink_mass, diag_mass. Rows are renormalized over valid keys, and
        metrics are averaged over valid query rows.
    """
    if attn.dim() != 3:
        raise ValueError(f'attn must have shape (num_heads, seq_len, seq_len), got {tuple(attn.shape)}')
    if mask.dim() != 1 or mask.size(0) != attn.size(-1):
        raise ValueError(f'mask must have shape ({attn.size(-1)},), got {tuple(mask.shape)}')
    valid_indices = mask.bool().nonzero(as_tuple=False).squeeze(-1)
    valid_len = int(valid_indices.numel())
    if valid_len == 0:
        raise ValueError('mask has no valid tokens')
    sub = attn.float().index_select(1, valid_indices).index_select(2, valid_indices)
    sub = sub / sub.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    num_heads = sub.size(0)
    positions = torch.arange(valid_len, device=sub.device)
    is_backward = positions.view(1, -1) > positions.view(-1, 1)
    backward_mass = (sub * is_backward.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
    distances = (positions.view(1, -1) - positions.view(-1, 1)).abs().float()
    mean_distance = ((sub * distances.unsqueeze(0)).sum(dim=-1) / float(max(valid_len - 1, 1))).mean(dim=-1)
    sink_mass = sub[:, :, 0].mean(dim=-1)
    diag_mass = sub.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    if valid_len > 1:
        entropy = (torch.special.entr(sub).sum(dim=-1) / math.log(valid_len)).mean(dim=-1)
    else:
        # All rows have effective length 1 and are excluded; report 0.0 to stay finite.
        entropy = torch.zeros(num_heads)
    per_head: dict[str, list[float]] = {
        'backward_mass': backward_mass.cpu().tolist(),
        'entropy': entropy.cpu().tolist(),
        'mean_distance': mean_distance.cpu().tolist(),
        'sink_mass': sink_mass.cpu().tolist(),
        'diag_mass': diag_mass.cpu().tolist(),
    }
    mean = {name: float(np.mean(values)) for name, values in per_head.items()}
    return {'per_head': per_head, 'mean': mean}


def position_contribution(token_states: torch.Tensor, mask: torch.Tensor, num_bins: int = 10) -> dict[str, Any]:
    """
    Measure how tokens at different relative positions contribute to the mean-pooled sentence embedding.

    Args:
        token_states: torch.Tensor of shape (seq_len, hidden_dim).
        mask: torch.Tensor of shape (seq_len,); True for valid tokens.
        num_bins: number of equal-width relative-position bins over valid tokens.

    Returns:
        dict with per-bin 'cos_mean' (cosine to the mean-pooled sentence embedding),
        'norm_share' (bin norm sum / total norm sum), 'count', and 'num_valid_tokens'.
        Empty bins report 0.0 with count 0 so callers can aggregate with count weights.
    """
    valid = mask.bool()
    states = token_states[valid].float()
    num_valid = states.size(0)
    if num_valid == 0:
        raise ValueError('mask has no valid tokens')
    sentence_emb = states.mean(dim=0)
    cos = F.cosine_similarity(states, sentence_emb.unsqueeze(0), dim=-1)
    norms = states.norm(dim=-1)
    total_norm = norms.sum().clamp_min(_EPS)
    ranks = torch.arange(num_valid, device=states.device)
    bin_ids = torch.div(ranks * num_bins, num_valid, rounding_mode='floor').clamp_max(num_bins - 1)
    cos_mean: list[float] = []
    norm_share: list[float] = []
    count: list[int] = []
    for bin_id in range(num_bins):
        selected = bin_ids == bin_id
        n_tokens = int(selected.sum())
        count.append(n_tokens)
        if n_tokens > 0:
            cos_mean.append(float(cos[selected].mean()))
            norm_share.append(float(norms[selected].sum() / total_norm))
        else:
            cos_mean.append(0.0)
            norm_share.append(0.0)
    return {'cos_mean': cos_mean, 'norm_share': norm_share, 'count': count, 'num_valid_tokens': num_valid}


def outlier_stats(embeddings: torch.Tensor, top_k: int = 8) -> dict[str, Any]:
    """
    Detect outlier (rogue) dimensions of an embedding matrix.

    Args:
        embeddings: torch.Tensor of shape (N, hidden_dim).
        top_k: number of top dimensions to report.

    Returns:
        dict with variance top-k dimension indices/values and shares, excess-kurtosis top-k
        dimension indices/values, and the mean contribution of the top-1 variance dimension
        to the squared embedding norm. Near-constant dimensions report excess kurtosis 0.0
        so that they never enter the kurtosis top-k.
    """
    emb = embeddings.float()
    hidden_dim = emb.size(1)
    k = min(top_k, hidden_dim)
    variance = emb.var(dim=0, unbiased=False)
    total_variance = variance.sum().clamp_min(_EPS)
    variance_top = torch.topk(variance, k)
    centered = emb - emb.mean(dim=0)
    m2 = centered.pow(2).mean(dim=0)
    m4 = centered.pow(4).mean(dim=0)
    excess_kurtosis = torch.where(m2 > _EPS, m4 / m2.clamp_min(_EPS).pow(2) - 3.0, torch.zeros_like(m2))
    kurtosis_top = torch.topk(excess_kurtosis, k)
    top1_dim = int(variance_top.indices[0])
    squared = emb.pow(2)
    top1_contribution = (squared[:, top1_dim] / squared.sum(dim=1).clamp_min(_EPS)).mean()
    return {
        'top_variance_dims': variance_top.indices.cpu().tolist(),
        'top_variance_values': variance_top.values.cpu().tolist(),
        'top1_variance_share': float(variance_top.values[0] / total_variance),
        'topk_variance_share': float(variance_top.values.sum() / total_variance),
        'top_kurtosis_dims': kurtosis_top.indices.cpu().tolist(),
        'top_kurtosis_values': kurtosis_top.values.cpu().tolist(),
        'top1_dim_mean_sq_norm_contribution': float(top1_contribution),
    }


def ablate_dims(embeddings: torch.Tensor, dims: list[int]) -> torch.Tensor:
    """
    Zero out the given dimensions of the embeddings (no renormalization).

    Args:
        embeddings: torch.Tensor of shape (N, hidden_dim).
        dims: dimension indices to zero out.

    Returns:
        A new tensor with the given dimensions set to 0; the input is left unchanged.
    """
    ablated = embeddings.clone()
    if dims:
        ablated[:, torch.as_tensor(dims, dtype=torch.long, device=ablated.device)] = 0
    return ablated


def effective_rank(embeddings: torch.Tensor) -> dict[str, float]:
    """
    Spectral summaries of a centered embedding matrix.

    Args:
        embeddings: torch.Tensor of shape (N, hidden_dim).

    Returns:
        dict with 'rankme' (exp of the entropy of normalized singular values),
        'top1_singular_share', 'top10_singular_share', and 'participation_ratio'
        ((sum sigma^2)^2 / sum sigma^4).
    """
    emb = embeddings.float()
    centered = emb - emb.mean(dim=0)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum().clamp_min(_EPS)
    normalized = singular_values / total
    rankme = float(torch.exp(torch.special.entr(normalized).sum()))
    squared = singular_values.pow(2)
    participation_ratio = float(squared.sum().pow(2) / squared.pow(2).sum().clamp_min(_EPS))
    return {
        'rankme': rankme,
        'top1_singular_share': float(singular_values[0] / total),
        'top10_singular_share': float(singular_values[:10].sum() / total),
        'participation_ratio': participation_ratio,
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Centered linear CKA between two representation matrices of shape (N, d_x) and (N, d_y).
    """
    if x.size(0) != y.size(0):
        raise ValueError(f'x and y must have the same number of rows, got {x.size(0)} and {y.size(0)}')
    x_centered = x.double() - x.double().mean(dim=0)
    y_centered = y.double() - y.double().mean(dim=0)
    hsic = x_centered.t().matmul(y_centered).norm(p='fro').pow(2)
    denom = x_centered.t().matmul(x_centered).norm(p='fro') * y_centered.t().matmul(y_centered).norm(p='fro')
    return float(hsic / denom.clamp_min(_EPS))


def mutual_knn_overlap(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    """
    Average overlap (shared neighbors / k) between per-point kNN sets in x-space and y-space.

    Neighbors are computed by cosine similarity, excluding each point itself.
    When k exceeds N - 1, both the kNN sets and the divisor shrink to N - 1.
    """
    if x.size(0) != y.size(0):
        raise ValueError(f'x and y must have the same number of rows, got {x.size(0)} and {y.size(0)}')
    num_points = x.size(0)
    if num_points <= 1:
        raise ValueError('mutual_knn_overlap requires at least 2 points')
    effective_k = min(k, num_points - 1)
    x_normalized = F.normalize(x.float(), dim=1)
    y_normalized = F.normalize(y.float(), dim=1)
    sim_x = x_normalized.matmul(x_normalized.t())
    sim_y = y_normalized.matmul(y_normalized.t())
    self_mask = torch.eye(num_points, dtype=torch.bool, device=sim_x.device)
    sim_x = sim_x.masked_fill(self_mask, float('-inf'))
    sim_y = sim_y.masked_fill(self_mask, float('-inf'))
    knn_x = sim_x.topk(effective_k, dim=1).indices
    knn_y = sim_y.topk(effective_k, dim=1).indices
    overlaps = [
        len(set(knn_x[i].cpu().tolist()) & set(knn_y[i].cpu().tolist())) / effective_k for i in range(num_points)
    ]
    return float(np.mean(overlaps))


def spearman(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """
    Spearman rank correlation between two 1-D sequences.
    """
    result = spearmanr(np.asarray(a), np.asarray(b))
    return float(result.statistic)


def bidirectional_forward_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Build an explicit 4D padding-only additive mask of shape (batch, 1, 1, seq_len).

    Valid key positions get 0, padded positions get the dtype minimum. Passing this
    precomputed 4D mask bypasses the causal-mask reconstruction in LlamaModel.forward,
    which is required because LlamaBiModel's bidirectional hooks only take effect on the
    flash-attention path of transformers 4.56 (the eager path would rebuild a causal mask).
    """
    min_value = torch.finfo(dtype).min
    return (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_value


def build_model_inputs(
    encoded: dict[str, torch.Tensor], needs_bidirectional_mask: bool, model_dtype: torch.dtype
) -> dict[str, Any]:
    """Return forward kwargs, replacing the 2D attention mask with a 4D bidirectional one when required."""
    inputs: dict[str, Any] = dict(encoded)
    if needs_bidirectional_mask:
        inputs['attention_mask'] = bidirectional_forward_mask(encoded['attention_mask'], model_dtype)
    return inputs


def ordered_text_sha256(texts: list[str]) -> str:
    """
    Order-sensitive SHA-256 over a text sequence (length-prefixed UTF-8 per item).

    Stored alongside layer-embedding dumps so that cross-model comparisons (CKA,
    mutual kNN) can verify both models encoded the identical sentence sequence.
    """
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode('utf-8')
        digest.update(str(len(encoded)).encode('ascii'))
        digest.update(b'\x00')
        digest.update(encoded)
    return digest.hexdigest()
