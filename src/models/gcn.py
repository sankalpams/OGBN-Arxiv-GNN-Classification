from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_embeddings: bool = False):
        embeddings = F.relu(self.conv1(x, edge_index))
        x = F.dropout(embeddings, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index)
        return (logits, embeddings) if return_embeddings else logits
