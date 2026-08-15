from __future__ import annotations

import torch


def node_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return total degree after edges have been made undirected."""
    return torch.bincount(edge_index[0], minlength=num_nodes)


def degree_statistics(degrees: torch.Tensor) -> dict[str, float]:
    return {
        "min_degree": float(degrees.min()),
        "max_degree": float(degrees.max()),
        "mean_degree": float(degrees.float().mean()),
        "median_degree": float(degrees.float().median()),
    }
