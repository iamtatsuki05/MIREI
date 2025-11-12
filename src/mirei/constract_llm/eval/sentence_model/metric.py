import torch
import torch.nn.functional as F


def compute_alignment(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """
    Compute alignment score between two sets of positive pairs.

    Args:
        z1: torch.Tensor of shape (batch_size, hidden_dim)
        z2: torch.Tensor of shape (batch_size, hidden_dim)

    Returns:
        float: Mean squared L2 distance between normalized pairs.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return ((z1 - z2).norm(p=2, dim=1) ** 2).mean().item()


def compute_uniformity(embeddings: torch.Tensor, t: float = 2.0) -> float:
    """
    Compute uniformity score for a set of embeddings.

    Args:
        embeddings: torch.Tensor of shape (N, hidden_dim)
        t: float, scaling parameter

    Returns:
        float: log of the mean of exp(-t * squared pairwise distances)
    """
    embeddings = F.normalize(embeddings, dim=1)  # (1) L2 normalization
    sq_norm = (embeddings**2).sum(dim=1, keepdim=True)  # (2) Squared norm for each vector
    dist_squared = sq_norm + sq_norm.T - 2 * torch.matmul(embeddings, embeddings.T)  # (3) Squared pairwise distances
    mask = ~torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)  # (4) Mask for off-diagonal
    exp_term = torch.exp(-t * dist_squared[mask])  # (5) exp(-t * dist^2)
    return torch.log(exp_term.mean()).item()  # (6) log(mean)
