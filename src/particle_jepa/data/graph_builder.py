from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data


class ParticleGraphBuilder:
    """Build a radius graph from particle positions and velocities."""

    def __init__(self, radius: float = 0.25, max_neighbors: int | None = None) -> None:
        self.radius = radius
        self.max_neighbors = max_neighbors

    def build(
        self,
        positions: Tensor,
        velocities: Tensor,
        particle_type: Tensor | None = None,
        material_type: Tensor | None = None,
        boundary: Tensor | None = None,
    ) -> Data:
        positions = positions.float()
        velocities = velocities.float()
        num_particles, spatial_dim = positions.shape
        device = positions.device

        particle_type = self._feature_or_zero(particle_type, num_particles, device)
        material_type = self._feature_or_zero(material_type, num_particles, device)
        boundary = self._feature_or_zero(boundary, num_particles, device)
        x = torch.cat([positions, velocities, particle_type, material_type, boundary], dim=-1)

        displacement = positions[:, None, :] - positions[None, :, :]
        distance = torch.linalg.norm(displacement, dim=-1)
        adjacency = (distance <= self.radius) & (distance > 0)
        edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()

        if self.max_neighbors is not None and edge_index.numel() > 0:
            edge_index = self._limit_neighbors(edge_index, distance, num_particles)

        if edge_index.numel() == 0:
            edge_attr = torch.empty((0, spatial_dim * 2 + 1), dtype=positions.dtype, device=device)
        else:
            src, dst = edge_index
            rel_pos = positions[dst] - positions[src]
            rel_vel = velocities[dst] - velocities[src]
            rel_dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
            edge_attr = torch.cat([rel_pos, rel_vel, rel_dist], dim=-1)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            velocity=velocities,
            particle_type=particle_type,
            material_type=material_type,
            boundary=boundary,
        )

    @staticmethod
    def _feature_or_zero(feature: Tensor | None, size: int, device: torch.device) -> Tensor:
        if feature is None:
            return torch.zeros((size, 1), dtype=torch.float32, device=device)
        if feature.ndim == 1:
            feature = feature[:, None]
        return feature.float().to(device)

    def _limit_neighbors(self, edge_index: Tensor, distance: Tensor, num_particles: int) -> Tensor:
        selected: list[Tensor] = []
        src_all, dst_all = edge_index
        for node in range(num_particles):
            mask = src_all == node
            if not torch.any(mask):
                continue
            candidates = edge_index[:, mask]
            dst = candidates[1]
            order = torch.argsort(distance[node, dst])[: self.max_neighbors]
            selected.append(candidates[:, order])
        if not selected:
            return edge_index[:, :0]
        return torch.cat(selected, dim=1).contiguous()
