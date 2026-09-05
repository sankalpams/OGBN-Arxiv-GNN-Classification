from __future__ import annotations

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm


class GCN(torch.nn.Module):
    """Spectral Graph Convolutional Network (GCN) with optional Normalization and Residual Connections."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        use_norm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.use_norm = use_norm
        self.use_residual = use_residual

        if use_norm:
            self.norm1 = BatchNorm(hidden_channels)
        else:
            self.norm1 = nn.Identity()

        if use_residual and in_channels != hidden_channels:
            self.residual_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.residual_proj = None

    def reset_parameters(self) -> None:
        """Resets all learnable parameters in convolutional and normalization layers."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if hasattr(self.norm1, "reset_parameters"):
            self.norm1.reset_parameters()
        if self.residual_proj is not None:
            self.residual_proj.reset_parameters()

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate hidden layer representations."""
        h = self.conv1(x, edge_index)
        if self.use_residual:
            res = self.residual_proj(x) if self.residual_proj is not None else x
            h = h + res
        h = self.norm1(h)
        return F.relu(h)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embeddings = self.get_embeddings(x, edge_index)
        h = F.dropout(embeddings, p=self.dropout, training=self.training)
        logits = self.conv2(h, edge_index)
        return (logits, embeddings) if return_embeddings else logits

