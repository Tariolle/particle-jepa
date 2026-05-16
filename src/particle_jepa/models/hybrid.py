from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from particle_jepa.models.decoders import AccelerationDecoder
from particle_jepa.models.encoders import ParticleGraphEncoder
from particle_jepa.models.predictors import LatentGraphPredictor


class HybridGNSJEPA(nn.Module):
    """Shared graph encoder with dynamics and latent future heads."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        message_passing_steps: int = 4,
        dropout: float = 0.0,
        mlp_layers: int = 2,
        max_horizon: int = 32,
        latent_predictor_steps: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ParticleGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
            mlp_layers=mlp_layers,
        )
        self.target_encoder = self.encoder
        self.dynamics_head = AccelerationDecoder(latent_dim, hidden_dim, mlp_layers=mlp_layers)
        self.horizon_embedding = nn.Embedding(max_horizon + 1, latent_dim)
        self.predictor = LatentGraphPredictor(
            latent_dim=latent_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            steps=latent_predictor_steps,
            dropout=dropout,
            mlp_layers=mlp_layers,
        )

    def forward(
        self, context_graph: Data | Batch, future_graph: Data | Batch, horizon: Tensor | None = None
    ) -> dict[str, Tensor]:
        node_latents, context_latent = self.encoder(context_graph)
        acceleration = self.dynamics_head(node_latents)
        target_node_latents, target_latent = self.target_encoder(future_graph)
        horizon = _resolve_horizon(
            context_graph, context_latent.size(0), context_latent.device, horizon
        )
        horizon_latent = self.horizon_embedding(horizon)
        node_prediction, prediction = self.predictor(
            node_latents,
            context_graph,
            horizon_latent,
        )
        return {
            "acceleration": acceleration,
            "prediction": prediction,
            "target": target_latent,
            "context": context_latent,
            "node_prediction": node_prediction,
            "node_target": target_node_latents,
            "node_context": node_latents,
        }

    @staticmethod
    def pool_nodes(node_latents: Tensor, graph: Data | Batch) -> Tensor:
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(node_latents.size(0), dtype=torch.long, device=node_latents.device)
        return global_mean_pool(node_latents, batch)


def _resolve_horizon(
    graph: Data | Batch, batch_size: int, device: torch.device, horizon: Tensor | None
) -> Tensor:
    if horizon is None:
        horizon = getattr(graph, "horizon", None)
    if horizon is None:
        horizon = torch.ones(batch_size, dtype=torch.long, device=device)
    horizon = horizon.to(device=device, dtype=torch.long).view(-1)
    if horizon.numel() == 1 and batch_size > 1:
        horizon = horizon.expand(batch_size)
    return horizon.clamp_min(0)
