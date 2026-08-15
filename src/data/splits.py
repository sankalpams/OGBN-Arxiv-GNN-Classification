from __future__ import annotations

import torch


def split_summary(split_idx: dict[str, torch.Tensor]) -> dict[str, int]:
    return {name: int(indices.numel()) for name, indices in split_idx.items()}
