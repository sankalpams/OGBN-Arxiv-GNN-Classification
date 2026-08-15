from __future__ import annotations

import torch


def neighbour_label_agreement(edge_index: torch.Tensor, labels: torch.Tensor, node_id: int) -> float:
    """Fraction of outgoing neighbours with the same class as a focal node."""
    neighbours = edge_index[1, edge_index[0] == node_id]
    if neighbours.numel() == 0:
        return 0.0
    labels = labels.squeeze()
    return float((labels[neighbours] == labels[node_id]).float().mean())
