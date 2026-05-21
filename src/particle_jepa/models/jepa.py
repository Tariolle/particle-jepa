from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from particle_jepa.models.encoders import ParticleGraphEncoder
from particle_jepa.models.predictors import (
    LatentGraphPredictor,
    LatentGraphTransformerPredictor,
)


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
        predictor_type: str = "message_passing",
        predictor_layers: int | None = None,
        predictor_heads: int = 4,
        predictor_dropout: float | None = None,
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
        predictor_dropout = dropout if predictor_dropout is None else predictor_dropout
        predictor_layers = predictor_layers or latent_predictor_steps
        if predictor_type in {"graph_transformer", "transformer"}:
            self.predictor = LatentGraphTransformerPredictor(
                latent_dim=latent_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                layers=predictor_layers,
                heads=predictor_heads,
                dropout=predictor_dropout,
                mlp_layers=mlp_layers,
            )
        elif predictor_type in {"message_passing", "mpnn"}:
            self.predictor = LatentGraphPredictor(
                latent_dim=latent_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                steps=latent_predictor_steps,
                dropout=dropout,
                mlp_layers=mlp_layers,
            )
        else:
            msg = f"Unsupported predictor_type '{predictor_type}'."
            raise ValueError(msg)

    def forward(
        self, context_graph: Data | Batch, future_graph: Data | Batch, horizon: Tensor | None = None
    ) -> dict[str, Tensor]:
        prediction_outputs = self.predict(context_graph, horizon=horizon)
        context_node_latents = prediction_outputs["node_context"]
        context_batch = _batch_vector(context_graph, context_node_latents)
        batch_size = _num_graphs(context_graph)
        target_node_latents, _ = self.target_encoder(future_graph, pool=False)
        target_latent = _mean_pool(target_node_latents, context_batch, batch_size)
        region_target = spatial_region_pool(
            target_node_latents,
            context_graph.pos,
            context_batch,
            self.region_grid_size,
            batch_size,
        )
        return {
            "prediction": prediction_outputs["prediction"],
            "target": target_latent,
            "context": prediction_outputs["context"],
            "node_prediction": prediction_outputs["node_prediction"],
            "node_target": target_node_latents,
            "node_context": context_node_latents,
            "region_prediction": prediction_outputs["region_prediction"],
            "region_context": prediction_outputs["region_context"],
            "region_target": region_target,
            "node_mask": getattr(context_graph, "dynamic_mask", None),
        }

    def predict(self, context_graph: Data | Batch, horizon: Tensor | None = None) -> dict:
        """Predict future latents from a context graph without using the target branch."""
        context_node_latents, _ = self.context_encoder(context_graph, pool=False)
        context_batch = _batch_vector(context_graph, context_node_latents)
        batch_size = _num_graphs(context_graph)
        context_latent = _mean_pool(context_node_latents, context_batch, batch_size)
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
        region_context = spatial_region_pool(
            context_node_latents,
            context_graph.pos,
            context_batch,
            self.region_grid_size,
            batch_size,
        )
        return {
            "prediction": prediction,
            "context": context_latent,
            "node_prediction": node_prediction,
            "node_context": context_node_latents,
            "region_prediction": region_prediction,
            "region_context": region_context,
        }

    def compile_regions(self, input_dtype: torch.dtype | None = None, **compile_kwargs) -> int:
        """Compile dense repeated regions while leaving sparse PyG graph glue eager."""
        return _compile_dense_regions(self, input_dtype=input_dtype, **compile_kwargs)


def _compile_dense_regions(
    module: nn.Module, input_dtype: torch.dtype | None = None, **compile_kwargs
) -> int:
    compiled = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential) and _is_shape_stable_mlp(child):
            compiled_child = torch.compile(child, **compile_kwargs)
            setattr(module, name, DtypeStableCompiledRegion(compiled_child, input_dtype))
            compiled += 1
        else:
            compiled += _compile_dense_regions(child, input_dtype=input_dtype, **compile_kwargs)
    return compiled


def _is_shape_stable_mlp(module: nn.Sequential) -> bool:
    linears = [child for child in module if isinstance(child, nn.Linear)]
    if not linears:
        return False
    return linears[0].in_features == linears[-1].out_features


class DtypeStableCompiledRegion(nn.Module):
    """Compiled dense region with an explicit floating input dtype contract."""

    def __init__(self, module: nn.Module, input_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.module = module
        self.input_dtype = input_dtype

    def forward(self, x: Tensor) -> Tensor:
        if self.input_dtype is not None and x.is_cuda and torch.is_floating_point(x):
            x = x.to(self.input_dtype)
        return self.module(x)


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
    values_fp32 = values.float()
    pooled = values_fp32.new_zeros((batch_size, values.size(-1)))
    counts = values_fp32.new_zeros((batch_size, 1))
    pooled.index_add_(0, batch, values_fp32)
    counts.index_add_(0, batch, values_fp32.new_ones((values.size(0), 1)))
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
    values_fp32 = values.float()
    pooled = values_fp32.new_zeros((batch_size * num_regions, values.size(-1)))
    counts = values_fp32.new_zeros((batch_size * num_regions, 1))
    pooled.index_add_(0, flat_region, values_fp32)
    counts.index_add_(0, flat_region, values_fp32.new_ones((values.size(0), 1)))
    pooled = pooled / counts.clamp_min(1.0)
    return pooled.view(batch_size, num_regions, values.size(-1))
