from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from particle_jepa.models.decoders import AccelerationDecoder
from particle_jepa.models.encoders import ParticleGraphEncoder
from particle_jepa.models.message_passing import make_mlp


class HybridGNSJEPA(nn.Module):
    """Shared graph encoder with dynamics and latent future heads."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        message_passing_steps: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ParticleGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
        )
        self.target_encoder = ParticleGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
        )
        self.dynamics_head = AccelerationDecoder(latent_dim, hidden_dim)
        self.predictor = make_mlp(latent_dim, hidden_dim, latent_dim, dropout)

    def forward(self, context_graph: Data | Batch, future_graph: Data | Batch) -> dict[str, Tensor]:
        node_latents, context_latent = self.encoder(context_graph)
        acceleration = self.dynamics_head(node_latents)
        with torch.no_grad():
            _, target_latent = self.target_encoder(future_graph)
        prediction = self.predictor(context_latent)
        return {
            "acceleration": acceleration,
            "prediction": prediction,
            "target": target_latent.detach(),
            "context": context_latent,
        }

    @staticmethod
    def pool_nodes(node_latents: Tensor, graph: Data | Batch) -> Tensor:
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(node_latents.size(0), dtype=torch.long, device=node_latents.device)
        return global_mean_pool(node_latents, batch)
