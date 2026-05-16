from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from particle_jepa.models.decoders import AccelerationDecoder
from particle_jepa.models.encoders import ParticleGraphEncoder


class GraphNetworkSimulator(nn.Module):
    """Compact GNS-style model that predicts particle acceleration."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        message_passing_steps: int = 4,
        dropout: float = 0.0,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ParticleGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
            mlp_layers=mlp_layers,
        )
        self.decoder = AccelerationDecoder(hidden_dim, hidden_dim, mlp_layers=mlp_layers)

    def forward(self, graph: Data | Batch) -> Tensor:
        node_latents, _ = self.encoder(graph)
        return self.decoder(node_latents)
