from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from particle_jepa.data.graph_builder import ParticleGraphBuilder


@dataclass(frozen=True)
class ToyParticleConfig:
    num_trajectories: int = 64
    num_particles: int = 48
    sequence_length: int = 24
    dimension: int = 2
    dt: float = 0.05
    future_offset: int = 4
    radius: float = 0.25
    noise_std: float = 0.002
    box_size: float = 1.0
    gravity: float = 0.0
    attraction_strength: float = 0.4
    repulsion_strength: float = 0.0
    num_particle_types: int = 2
    seed: int = 7


def generate_toy_rollouts(config: ToyParticleConfig) -> dict:
    """Generate bounded 2D particle rollouts for smoke testing."""
    if config.dimension != 2:
        msg = "ToyParticleDataset currently supports dimension=2."
        raise ValueError(msg)
    generator = torch.Generator().manual_seed(config.seed)
    positions = torch.rand(
        config.num_trajectories,
        config.sequence_length,
        config.num_particles,
        config.dimension,
        generator=generator,
    )
    velocities = 0.15 * torch.randn(
        config.num_trajectories,
        config.sequence_length,
        config.num_particles,
        config.dimension,
        generator=generator,
    )
    particle_types = torch.randint(
        low=0,
        high=config.num_particle_types,
        size=(config.num_trajectories, config.num_particles),
        generator=generator,
    )
    positions[:, 0] *= config.box_size

    center = torch.tensor([config.box_size / 2, config.box_size / 2])
    gravity = torch.tensor([0.0, -config.gravity])
    for t in range(1, config.sequence_length):
        prev_pos = positions[:, t - 1]
        prev_vel = velocities[:, t - 1]
        attraction = config.attraction_strength * (center - prev_pos)
        swirl = torch.stack([-prev_vel[..., 1], prev_vel[..., 0]], dim=-1) * 0.15
        repulsion = _short_range_repulsion(prev_pos, config.repulsion_strength)
        noise = config.noise_std * torch.randn(prev_vel.shape, generator=generator)
        vel = 0.985 * prev_vel + config.dt * (attraction + swirl + repulsion + gravity) + noise
        pos = prev_pos + config.dt * vel

        low = pos < 0.0
        high = pos > config.box_size
        vel = torch.where(low | high, -0.65 * vel, vel)
        pos = pos.clamp(0.0, config.box_size)
        positions[:, t] = pos
        velocities[:, t] = vel

    return {
        "positions": positions,
        "velocities": velocities,
        "particle_types": particle_types,
        "metadata": {
            "dt": config.dt,
            "box_size": config.box_size,
            "dimension": config.dimension,
            "num_particle_types": config.num_particle_types,
        },
    }


def generate_toy_trajectories(config: ToyParticleConfig) -> tuple[Tensor, Tensor]:
    """Generate positions and velocities for backward-compatible callers."""
    rollout = generate_toy_rollouts(config)
    return rollout["positions"], rollout["velocities"]


def _short_range_repulsion(positions: Tensor, strength: float) -> Tensor:
    if strength == 0.0:
        return torch.zeros_like(positions)
    delta = positions[:, :, None, :] - positions[:, None, :, :]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-4)
    mask = (distance < 0.08).float()
    return strength * (delta / distance.pow(2) * mask).sum(dim=2)


class ToyParticleDataset(Dataset):
    """Dataset returning graph pairs at `t` and `t + future_offset`."""

    def __init__(self, config: ToyParticleConfig | None = None) -> None:
        self.config = config or ToyParticleConfig()
        self.trajectory = generate_toy_rollouts(self.config)
        self.positions = self.trajectory["positions"]
        self.velocities = self.trajectory["velocities"]
        self.particle_types = self.trajectory["particle_types"]
        self.graph_builder = ParticleGraphBuilder(radius=self.config.radius)
        self.samples_per_trajectory = self.config.sequence_length - self.config.future_offset

    def __len__(self) -> int:
        return self.config.num_trajectories * self.samples_per_trajectory

    def __getitem__(self, index: int):
        traj_idx = index // self.samples_per_trajectory
        time_idx = index % self.samples_per_trajectory
        future_idx = time_idx + self.config.future_offset

        context = self.graph_builder.build(
            self.positions[traj_idx, time_idx],
            self.velocities[traj_idx, time_idx],
            particle_type=self.particle_types[traj_idx],
        )
        future = self.graph_builder.build(
            self.positions[traj_idx, future_idx],
            self.velocities[traj_idx, future_idx],
            particle_type=self.particle_types[traj_idx],
        )

        next_position = self.positions[traj_idx, time_idx + 1]
        next_velocity = self.velocities[traj_idx, time_idx + 1]
        acceleration = (next_velocity - context.velocity) / self.config.dt

        context.y_pos = next_position
        context.y_velocity = next_velocity
        context.y_acceleration = acceleration
        context.horizon = torch.tensor([self.config.future_offset], dtype=torch.long)
        future.horizon = torch.tensor([self.config.future_offset], dtype=torch.long)
        return context, future
