from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from particle_jepa.models.message_passing import GraphProcessor, make_mlp


class ParticleGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        message_passing_steps: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_encoder = make_mlp(node_dim, hidden_dim, hidden_dim, dropout)
        self.edge_encoder = make_mlp(edge_dim, hidden_dim, hidden_dim, dropout)
        self.processor = GraphProcessor(hidden_dim, message_passing_steps, dropout)
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, graph: Data | Batch) -> tuple[Tensor, Tensor]:
        x = self.node_encoder(graph.x)
        edge_attr = self.edge_encoder(graph.edge_attr)
        x, _ = self.processor(x, graph.edge_index, edge_attr)
        node_latents = self.projection(x)
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_latent = global_mean_pool(node_latents, batch)
        return node_latents, graph_latent
