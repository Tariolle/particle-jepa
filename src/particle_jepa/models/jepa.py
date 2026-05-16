from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from particle_jepa.models.encoders import ParticleGraphEncoder
from particle_jepa.models.message_passing import make_mlp


class ParticleJEPA(nn.Module):
    """Predict a future graph latent from a current graph latent."""

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
        self.context_encoder = ParticleGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = make_mlp(latent_dim, hidden_dim, latent_dim, dropout)

    def forward(self, context_graph: Data | Batch, future_graph: Data | Batch) -> dict[str, Tensor]:
        _, context_latent = self.context_encoder(context_graph)
        with torch.no_grad():
            _, target_latent = self.target_encoder(future_graph)
        prediction = self.predictor(context_latent)
        return {
            "prediction": prediction,
            "target": target_latent.detach(),
            "context": context_latent,
        }
