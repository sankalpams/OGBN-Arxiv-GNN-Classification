from __future__ import annotations

from pathlib import Path
import torch
from torch_geometric.data import Data


def standardize_features(x: torch.Tensor) -> torch.Tensor:
    """Column-wise z-score standardisation with numerical protection."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std


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
