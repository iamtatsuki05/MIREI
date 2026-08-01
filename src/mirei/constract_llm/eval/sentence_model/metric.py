import torch
import torch.nn.functional as F


def compute_alignment_sq_distances(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pair squared L2 distances between normalized positive pairs.

    Args:
        z1: torch.Tensor of shape (batch_size, hidden_dim)
        z2: torch.Tensor of shape (batch_size, hidden_dim)

    Returns:
        torch.Tensor of shape (batch_size,): squared L2 distance per pair.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return (z1 - z2).norm(p=2, dim=1) ** 2


def compute_alignment(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """
    Compute alignment score between two sets of positive pairs.

    Args:
        z1: torch.Tensor of shape (batch_size, hidden_dim)
        z2: torch.Tensor of shape (batch_size, hidden_dim)

    Returns:
        float: Mean squared L2 distance between normalized pairs.
    """
    return compute_alignment_sq_distances(z1, z2).mean().item()


def compute_pairwise_sq_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute off-diagonal squared pairwise distances of normalized embeddings.

    Args:
        embeddings: torch.Tensor of shape (N, hidden_dim)

    Returns:
        torch.Tensor of shape (N * (N - 1),): squared L2 distances for all ordered off-diagonal pairs.
    """
    embeddings = F.normalize(embeddings, dim=1)  # (1) L2 normalization
    sq_norm = (embeddings**2).sum(dim=1, keepdim=True)  # (2) Squared norm for each vector
    dist_squared = sq_norm + sq_norm.T - 2 * torch.matmul(embeddings, embeddings.T)  # (3) Squared pairwise distances
    mask = ~torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)  # (4) Mask for off-diagonal
    return dist_squared[mask]


def compute_uniformity_from_sq_distances(sq_distances: torch.Tensor, t: float = 2.0) -> float:
    """
    Compute uniformity score from precomputed squared pairwise distances.

    Args:
        sq_distances: torch.Tensor of squared pairwise distances
        t: float, scaling parameter

    Returns:
        float: log of the mean of exp(-t * squared pairwise distances)
    """
    return torch.log(torch.exp(-t * sq_distances).mean()).item()


def compute_uniformity(embeddings: torch.Tensor, t: float = 2.0) -> float:
    """
    Compute uniformity score for a set of embeddings.

    Args:
        embeddings: torch.Tensor of shape (N, hidden_dim)
        t: float, scaling parameter

    Returns:
        float: log of the mean of exp(-t * squared pairwise distances)
    """
    return compute_uniformity_from_sq_distances(compute_pairwise_sq_distances(embeddings), t=t)
