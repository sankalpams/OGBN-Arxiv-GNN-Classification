from __future__ import annotations

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def connected_component_summary(data: Data, max_nodes: int | None = None) -> dict[str, int]:
    """Analyse connected components; optionally restrict to a sampled prefix for speed."""
    if max_nodes is not None:
        data = data.subgraph(range(min(max_nodes, data.num_nodes)))
    graph = to_networkx(data, to_undirected=True)
    sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    return {"component_count": len(sizes), "largest_component_size": sizes[0] if sizes else 0}
