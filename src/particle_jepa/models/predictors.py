from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter, softmax

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


class LatentGraphTransformerPredictor(nn.Module):
    """Edge-aware graph transformer for latent particle dynamics."""

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        if latent_dim % heads != 0:
            msg = f"latent_dim={latent_dim} must be divisible by heads={heads}."
            raise ValueError(msg)
        self.node_input = make_mlp(latent_dim * 2, hidden_dim, latent_dim, dropout, mlp_layers)
        self.layers = nn.ModuleList(
            [
                EdgeAwareGraphTransformerLayer(
                    latent_dim=latent_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    mlp_layers=mlp_layers,
                )
                for _ in range(layers)
            ]
        )
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
        for layer in self.layers:
            x = layer(x, graph.edge_index, graph.edge_attr)
        node_prediction = self.node_output(x)
        graph_prediction = _mean_pool(
            node_prediction, node_batch, int(getattr(graph, "num_graphs", 1))
        )
        return node_prediction, graph_prediction


class EdgeAwareGraphTransformerLayer(nn.Module):
    """Pre-norm graph transformer block with edge-conditioned attention."""

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        mlp_layers: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = latent_dim // heads
        self.attn_norm = nn.LayerNorm(latent_dim)
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.edge_key = make_mlp(edge_dim, hidden_dim, latent_dim, dropout, mlp_layers)
        self.edge_value = make_mlp(edge_dim, hidden_dim, latent_dim, dropout, mlp_layers)
        self.edge_bias = make_mlp(edge_dim, hidden_dim, heads, dropout, mlp_layers)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.ffn = make_mlp(latent_dim, hidden_dim, latent_dim, dropout, mlp_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return x + self.dropout(self.ffn(self.ffn_norm(x)))

        source, target = edge_index
        x_norm = self.attn_norm(x)
        q = self.q_proj(x_norm).view(-1, self.heads, self.head_dim)
        k = self.k_proj(x_norm).view(-1, self.heads, self.head_dim)
        v = self.v_proj(x_norm).view(-1, self.heads, self.head_dim)
        edge_key = self.edge_key(edge_attr).view(-1, self.heads, self.head_dim)
        edge_value = self.edge_value(edge_attr).view(-1, self.heads, self.head_dim)
        edge_bias = self.edge_bias(edge_attr)

        logits = (q[target] * (k[source] + edge_key)).sum(dim=-1)
        logits = logits / math.sqrt(self.head_dim) + edge_bias
        attention = softmax(logits.float(), target, num_nodes=x.size(0)).to(dtype=x.dtype)
        messages = (v[source] + edge_value) * attention.unsqueeze(-1)
        messages = messages.reshape(edge_attr.size(0), self.heads * self.head_dim)
        aggregated = scatter(messages, target, dim=0, dim_size=x.size(0), reduce="sum")
        x = x + self.dropout(self.out_proj(aggregated))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


def _node_batch(graph: Data | Batch, node_latents: Tensor) -> Tensor:
    batch = getattr(graph, "batch", None)
    if batch is None:
        return torch.zeros(node_latents.size(0), dtype=torch.long, device=node_latents.device)
    return batch


def _mean_pool(values: Tensor, batch: Tensor, batch_size: int) -> Tensor:
    values_fp32 = values.float()
    pooled = values_fp32.new_zeros((batch_size, values.size(-1)))
    counts = values_fp32.new_zeros((batch_size, 1))
    pooled.index_add_(0, batch, values_fp32)
    counts.index_add_(0, batch, values_fp32.new_ones((values.size(0), 1)))
    return pooled / counts.clamp_min(1.0)
