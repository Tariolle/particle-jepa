from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from particle_jepa.models.message_passing import GraphProcessor, make_mlp


class LatentGraphPredictor(nn.Module):
    """Message-passing predictor for latent graph dynamics."""

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        steps: int = 2,
        dropout: float = 0.0,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.edge_encoder = make_mlp(edge_dim, hidden_dim, latent_dim, dropout, mlp_layers)
        self.node_input = make_mlp(latent_dim * 2, hidden_dim, latent_dim, dropout, mlp_layers)
        self.processor = GraphProcessor(latent_dim, steps, dropout, mlp_layers)
        self.node_output = make_mlp(latent_dim, hidden_dim, latent_dim, dropout, mlp_layers)

    def forward(
        self,
        node_latents: Tensor,
        graph: Data | Batch,
        horizon_latent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        node_batch = _node_batch(graph, node_latents)
        node_horizon = horizon_latent[node_batch]
        x = self.node_input(torch.cat([node_latents, node_horizon], dim=-1))
        edge_attr = self.edge_encoder(graph.edge_attr)
        x, _ = self.processor(x, graph.edge_index, edge_attr)
        node_prediction = self.node_output(x)
        graph_prediction = _mean_pool(
            node_prediction, node_batch, int(getattr(graph, "num_graphs", 1))
        )
        return node_prediction, graph_prediction


def _node_batch(graph: Data | Batch, node_latents: Tensor) -> Tensor:
    batch = getattr(graph, "batch", None)
    if batch is None:
        return torch.zeros(node_latents.size(0), dtype=torch.long, device=node_latents.device)
    return batch


def _mean_pool(values: Tensor, batch: Tensor, batch_size: int) -> Tensor:
    pooled = values.new_zeros((batch_size, values.size(-1)))
    counts = values.new_zeros((batch_size, 1))
    pooled.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones((values.size(0), 1), device=values.device))
    return pooled / counts.clamp_min(1.0)
