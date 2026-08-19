from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes weighted multi-class classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes unweighted macro multi-class classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def top_k_accuracy(y_true: np.ndarray | torch.Tensor, logits: np.ndarray | torch.Tensor, k: int = 5) -> float:
    """Computes Top-K classification accuracy."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    
    y_true = y_true.view(-1, 1)
    _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = (top_k_preds == y_true).any(dim=1).float().mean().item()
    return round(float(correct), 4)


def compute_benchmark_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Generates a complete benchmark summary covering weighted, macro, and top-k accuracy."""
    summary = classification_metrics(y_true, y_pred)
    summary.update(macro_metrics(y_true, y_pred))
    if logits is not None:
        summary["top_3_accuracy"] = top_k_accuracy(y_true, logits, k=3)
        summary["top_5_accuracy"] = top_k_accuracy(y_true, logits, k=5)
    return summary

