from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


def make_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class InteractionNetworkLayer(MessagePassing):
    """Interaction Network block with edge and node residual updates."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__(aggr="add")
        self.edge_mlp = make_mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.node_mlp = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        edge_attr = edge_attr + edge_update
        aggregated = scatter(edge_attr, row, dim=0, dim_size=x.size(0), reduce="sum")
        node_update = self.node_mlp(torch.cat([x, aggregated], dim=-1))
        return x + node_update, edge_attr


class GraphProcessor(nn.Module):
    def __init__(self, hidden_dim: int, steps: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [InteractionNetworkLayer(hidden_dim, dropout=dropout) for _ in range(steps)]
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr
