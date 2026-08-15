from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_embeddings: bool = False):
        embeddings = F.elu(self.conv1(x, edge_index))
        x = F.dropout(embeddings, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index)
        return (logits, embeddings) if return_embeddings else logits

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Return first-layer edge attention for explainability."""
        _, (edges, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return edges, alpha
