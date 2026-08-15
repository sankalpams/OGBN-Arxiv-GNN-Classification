from __future__ import annotations

import torch


def attention_summary(alpha: torch.Tensor) -> dict[str, float]:
    """Summarise GAT attention coefficients across heads and edges."""
    return {"mean_attention": float(alpha.mean()), "max_attention": float(alpha.max()), "min_attention": float(alpha.min())}
