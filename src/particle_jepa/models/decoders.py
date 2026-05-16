from __future__ import annotations

from torch import Tensor, nn

from particle_jepa.models.message_passing import make_mlp


class AccelerationDecoder(nn.Module):
    def __init__(
        self, latent_dim: int, hidden_dim: int = 128, spatial_dim: int = 2, mlp_layers: int = 2
    ) -> None:
        super().__init__()
        self.net = make_mlp(latent_dim, hidden_dim, spatial_dim, num_layers=mlp_layers)

    def forward(self, node_latents: Tensor) -> Tensor:
        return self.net(node_latents)
