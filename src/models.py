"""Node-classification architectures: GCN, GraphSAGE, and MLP."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, SAGEConv


class GCNNet(nn.Module):
    """Two-layer GCN, following the PyG introduction tutorial."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGENet(nn.Module):
    """Two-layer GraphSAGE (mean aggregation)."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class MLPNet(nn.Module):
    """Two-layer MLP baseline that ignores graph edges."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


def build_model(
    gnn_type: str, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
) -> nn.Module:
    if gnn_type == "gcn":
        return GCNNet(in_channels, hidden_channels, out_channels, dropout)
    if gnn_type == "sage":
        return GraphSAGENet(in_channels, hidden_channels, out_channels, dropout)
    if gnn_type == "mlp":
        return MLPNet(in_channels, hidden_channels, out_channels, dropout)
    raise ValueError(f"Unsupported model type '{gnn_type}' (choose 'gcn', 'sage', or 'mlp').")
