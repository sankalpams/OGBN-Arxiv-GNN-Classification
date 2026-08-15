from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from .metrics import classification_metrics


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, data: Data, index: torch.Tensor, split_name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    predictions = model(data.x, data.edge_index).argmax(dim=-1)
    y_true = data.y.squeeze()[index].detach().cpu().numpy()
    y_pred = predictions[index].detach().cpu().numpy()
    return {"split": split_name, **classification_metrics(y_true, y_pred)}, y_true, y_pred
