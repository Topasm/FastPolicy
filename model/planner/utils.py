import torch


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Convert raw critic scores to probabilities via softmax.
    """
    return torch.softmax(scores, dim=0)
