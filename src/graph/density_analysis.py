from __future__ import annotations


def graph_density(num_nodes: int, num_edges: int, directed: bool = False) -> float:
    """Compute simple-graph density, excluding possible self loops."""
    if num_nodes < 2:
        return 0.0
    possible_edges = num_nodes * (num_nodes - 1) if directed else num_nodes * (num_nodes - 1) / 2
    return float(num_edges / possible_edges)
