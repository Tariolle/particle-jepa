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
        edge_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        message_passing_steps: int = 4,
        dropout: float = 0.0,
        mlp_layers: int = 2,
        max_horizon: int = 32,
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
        self.target_encoder = ParticleGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
            mlp_layers=mlp_layers,
        )
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.dynamics_head = AccelerationDecoder(latent_dim, hidden_dim, mlp_layers=mlp_layers)
        self.horizon_embedding = nn.Embedding(max_horizon + 1, latent_dim)
        self.predictor = make_mlp(latent_dim * 2, hidden_dim, latent_dim, dropout, mlp_layers)

    def forward(
        self, context_graph: Data | Batch, future_graph: Data | Batch, horizon: Tensor | None = None
    ) -> dict[str, Tensor]:
        node_latents, context_latent = self.encoder(context_graph)
        acceleration = self.dynamics_head(node_latents)
        with torch.no_grad():
            _, target_latent = self.target_encoder(future_graph)
        horizon = _resolve_horizon(
            context_graph, context_latent.size(0), context_latent.device, horizon
        )
        horizon_latent = self.horizon_embedding(horizon)
        prediction = self.predictor(torch.cat([context_latent, horizon_latent], dim=-1))
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

    @torch.no_grad()
    def update_target_encoder(self, decay: float = 0.99) -> None:
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)


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
