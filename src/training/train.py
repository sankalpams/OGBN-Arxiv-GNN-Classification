from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch_geometric.data import Data
from .optimizer import build_optimizer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(model: torch.nn.Module, data: Data, index: torch.Tensor, evaluator: Evaluator) -> float:
    model.eval()
    predictions = model(data.x, data.edge_index).argmax(dim=-1, keepdim=True)
    return float(evaluator.eval({"y_true": data.y[index], "y_pred": predictions[index]})["acc"])


def fit(model: torch.nn.Module, data: Data, split_idx: dict, epochs: int = 200,
        learning_rate: float = 0.01, weight_decay: float = 5e-4,
        checkpoint_path: str | Path | None = None) -> pd.DataFrame:
    """Full-batch train a GNN and retain its best validation checkpoint."""
    evaluator = Evaluator(name="ogbn-arxiv")
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    best_state, best_valid = None, -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[split_idx["train"]], data.y.squeeze()[split_idx["train"]])
        loss.backward()
        optimizer.step()

        train_acc = accuracy(model, data, split_idx["train"], evaluator)
        valid_acc = accuracy(model, data, split_idx["valid"], evaluator)
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "train_accuracy": train_acc, "validation_accuracy": valid_acc})
        if valid_acc > best_valid:
            best_valid, best_state = valid_acc, deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
    return pd.DataFrame(history)
