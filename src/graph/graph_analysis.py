from __future__ import annotations

from torch_geometric.data import Data
from .degree_analysis import degree_statistics, node_degrees
from .density_analysis import graph_density


def graph_summary(data: Data) -> dict[str, float | int]:
    degrees = node_degrees(data.edge_index, data.num_nodes)
    return {
        "nodes": int(data.num_nodes),
        "edges": int(data.num_edges),
        "features": int(data.num_features),
        "density": graph_density(data.num_nodes, data.num_edges),
        **degree_statistics(degrees),
    }
