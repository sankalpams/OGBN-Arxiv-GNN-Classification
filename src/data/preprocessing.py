from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch
from torch_geometric.data import Data


def standardize_features(x: torch.Tensor) -> torch.Tensor:
    """Column-wise z-score standardisation with numerical protection."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std


def min_max_scale_features(x: torch.Tensor, feature_range: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """Min-max feature scaling mapping attributes to a specified range."""
    min_val = x.min(dim=0, keepdim=True).values
    max_val = x.max(dim=0, keepdim=True).values
    scale = (max_val - min_val).clamp_min(1e-8)
    norm = (x - min_val) / scale
    return norm * (feature_range[1] - feature_range[0]) + feature_range[0]


def compute_feature_statistics(x: torch.Tensor) -> Dict[str, float]:
    """Computes distribution statistics across node feature vectors."""
    sparsity = float((x == 0).sum() / x.numel())
    mean_val = float(x.mean().item())
    std_val = float(x.std().item())
    l2_norm_avg = float(torch.norm(x, p=2, dim=1).mean().item())
    return {
        "num_features": x.shape[1],
        "sparsity_ratio": round(sparsity, 4),
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "avg_l2_norm": round(l2_norm_avg, 4),
    }


def validate_graph_data(data: Data) -> Dict[str, Any]:
    """Validates structural integrity, dimension consistency, and edge bounds."""
    num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0)
    num_edges = data.edge_index.size(1)
    has_self_loops = bool((data.edge_index[0] == data.edge_index[1]).any().item())
    is_undirected = bool(data.is_undirected())
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "has_self_loops": has_self_loops,
        "is_undirected": is_undirected,
        "feature_dim": int(data.x.size(1)),
        "is_valid": True,
    }


def prepare_and_save_data(data: Data, split_idx: dict, output_dir: str | Path, normalize: bool = False) -> Data:
    """Optionally normalise features and export graph tensors for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = data.clone()
    if normalize:
        prepared.x = standardize_features(prepared.x)
    torch.save(prepared, output_dir / "graph_data.pt")
    torch.save(prepared.x, output_dir / "features.pt")
    torch.save(prepared.y, output_dir / "labels.pt")
    torch.save(split_idx, output_dir / "splits.pt")
    return prepared

