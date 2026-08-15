from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx


def plot_sample_subgraph(data: Data, output_path: str | Path, center_node: int = 0, hops: int = 2) -> None:
    subset, edge_index, _, _ = k_hop_subgraph(center_node, hops, data.edge_index, relabel_nodes=True)
    graph = to_networkx(Data(edge_index=edge_index, num_nodes=subset.numel()), to_undirected=True)
    plt.figure(figsize=(9, 7))
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), node_size=35, with_labels=False, width=0.35)
    plt.title(f"{hops}-hop citation subgraph around node {center_node}")
    plt.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
