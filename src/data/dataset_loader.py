from __future__ import annotations

import torch
from pathlib import Path
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected

# PyTorch >=2.6 changed torch.load default to weights_only=True.
# OGB's processed .pt files contain PyG-specific globals that need to be
# explicitly allowed, or we patch torch.load to use weights_only=False.
try:
    from torch_geometric.data.data import DataEdgeAttr, GlobalStorage
    from torch_geometric.data import HeteroData
    _safe_globals = [DataEdgeAttr, GlobalStorage, HeteroData]
    torch.serialization.add_safe_globals(_safe_globals)
except Exception:
    pass  # older PyG versions don't need this

# Fallback: monkey-patch torch.load to always use weights_only=False for OGB
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load


def load_ogbn_arxiv(root: str | Path, make_undirected: bool = True) -> tuple[PygNodePropPredDataset, Data, dict]:
    """Download/load OGBN-Arxiv and return graph data with official OGB splits."""
    transform = ToUndirected() if make_undirected else None
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root), transform=transform)
    return dataset, dataset[0], dataset.get_idx_split()

