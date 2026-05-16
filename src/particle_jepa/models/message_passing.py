from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float = 0.0,
    num_layers: int = 2,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(max(num_layers - 1, 1)):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class InteractionNetworkLayer(MessagePassing):
    """Interaction Network block with edge and node residual updates."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0, mlp_layers: int = 2) -> None:
        super().__init__(aggr="add")
        self.edge_mlp = make_mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout, mlp_layers)
        self.node_mlp = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout, mlp_layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        edge_attr = edge_attr + edge_update
        aggregated = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce="sum")
        node_update = self.node_mlp(torch.cat([x, aggregated], dim=-1))
        return x + node_update, edge_attr


class GraphProcessor(nn.Module):
    def __init__(
        self, hidden_dim: int, steps: int, dropout: float = 0.0, mlp_layers: int = 2
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                InteractionNetworkLayer(hidden_dim, dropout=dropout, mlp_layers=mlp_layers)
                for _ in range(steps)
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr
