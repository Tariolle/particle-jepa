from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from particle_jepa.models.encoders import ParticleGraphEncoder
from particle_jepa.models.predictors import LatentGraphPredictor


class ParticleJEPA(nn.Module):
    """Predict a future graph latent from a current graph latent."""

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
        region_grid_size: int = 4,
    ) -> None:
        super().__init__()
        self.region_grid_size = region_grid_size
        self.context_encoder = ParticleGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
            mlp_layers=mlp_layers,
        )
        self.target_encoder = self.context_encoder
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
        context_node_latents, _ = self.context_encoder(context_graph, pool=False)
        target_node_latents, _ = self.target_encoder(future_graph, pool=False)
        context_batch = _batch_vector(context_graph, context_node_latents)
        batch_size = _num_graphs(context_graph)
        context_latent = _mean_pool(context_node_latents, context_batch, batch_size)
        target_latent = _mean_pool(target_node_latents, context_batch, batch_size)
        horizon = _resolve_horizon(
            context_graph, context_latent.size(0), context_latent.device, horizon
        )
        horizon_latent = self.horizon_embedding(horizon)
        node_prediction, prediction = self.predictor(
            context_node_latents,
            context_graph,
            horizon_latent,
        )
        region_prediction = spatial_region_pool(
            node_prediction,
            context_graph.pos,
            context_batch,
            self.region_grid_size,
            batch_size,
        )
        region_target = spatial_region_pool(
            target_node_latents,
            context_graph.pos,
            context_batch,
            self.region_grid_size,
            batch_size,
        )
        return {
            "prediction": prediction,
            "target": target_latent,
            "context": context_latent,
            "node_prediction": node_prediction,
            "node_target": target_node_latents,
            "node_context": context_node_latents,
            "region_prediction": region_prediction,
            "region_target": region_target,
            "node_mask": getattr(context_graph, "dynamic_mask", None),
        }


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


def _batch_vector(graph: Data | Batch, node_latents: Tensor) -> Tensor:
    batch = getattr(graph, "batch", None)
    if batch is None:
        return torch.zeros(node_latents.size(0), dtype=torch.long, device=node_latents.device)
    return batch


def _num_graphs(graph: Data | Batch) -> int:
    return int(getattr(graph, "num_graphs", 1))


def _mean_pool(values: Tensor, batch: Tensor, batch_size: int) -> Tensor:
    pooled = values.new_zeros((batch_size, values.size(-1)))
    counts = values.new_zeros((batch_size, 1))
    pooled.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones((values.size(0), 1), device=values.device))
    return pooled / counts.clamp_min(1.0)


def spatial_region_pool(
    values: Tensor,
    positions: Tensor,
    batch: Tensor,
    grid_size: int,
    batch_size: int,
) -> Tensor:
    """Pool node latents into fixed spatial bins using current particle positions."""
    if positions.size(-1) < 2:
        msg = "spatial_region_pool expects at least 2D particle positions."
        raise ValueError(msg)
    num_regions = grid_size * grid_size
    xy = positions[:, :2].clamp(0.0, 1.0 - 1e-6)
    bins = (xy * grid_size).long().clamp(0, grid_size - 1)
    region = bins[:, 1] * grid_size + bins[:, 0]
    flat_region = batch * num_regions + region
    pooled = values.new_zeros((batch_size * num_regions, values.size(-1)))
    counts = values.new_zeros((batch_size * num_regions, 1))
    pooled.index_add_(0, flat_region, values)
    counts.index_add_(0, flat_region, torch.ones((values.size(0), 1), device=values.device))
    pooled = pooled / counts.clamp_min(1.0)
    return pooled.view(batch_size, num_regions, values.size(-1))
