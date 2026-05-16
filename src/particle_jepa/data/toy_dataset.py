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
    dt: float = 0.05
    future_offset: int = 4
    radius: float = 0.25
    noise_std: float = 0.002
    box_size: float = 1.0
    seed: int = 7


def generate_toy_trajectories(config: ToyParticleConfig) -> tuple[Tensor, Tensor]:
    """Generate simple bounded particle trajectories with weak pairwise attraction."""
    generator = torch.Generator().manual_seed(config.seed)
    positions = torch.rand(
        config.num_trajectories,
        config.sequence_length,
        config.num_particles,
        2,
        generator=generator,
    )
    velocities = 0.15 * torch.randn(
        config.num_trajectories,
        config.sequence_length,
        config.num_particles,
        2,
        generator=generator,
    )
    positions[:, 0] *= config.box_size

    center = torch.tensor([config.box_size / 2, config.box_size / 2])
    for t in range(1, config.sequence_length):
        prev_pos = positions[:, t - 1]
        prev_vel = velocities[:, t - 1]
        attraction = 0.4 * (center - prev_pos)
        swirl = torch.stack([-prev_vel[..., 1], prev_vel[..., 0]], dim=-1) * 0.15
        noise = config.noise_std * torch.randn(prev_vel.shape, generator=generator)
        vel = 0.985 * prev_vel + config.dt * (attraction + swirl) + noise
        pos = prev_pos + config.dt * vel

        low = pos < 0.0
        high = pos > config.box_size
        vel = torch.where(low | high, -0.65 * vel, vel)
        pos = pos.clamp(0.0, config.box_size)
        positions[:, t] = pos
        velocities[:, t] = vel

    return positions, velocities


class ToyParticleDataset(Dataset):
    """Dataset returning graph pairs at `t` and `t + future_offset`."""

    def __init__(self, config: ToyParticleConfig | None = None) -> None:
        self.config = config or ToyParticleConfig()
        self.positions, self.velocities = generate_toy_trajectories(self.config)
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
        )
        future = self.graph_builder.build(
            self.positions[traj_idx, future_idx],
            self.velocities[traj_idx, future_idx],
        )

        next_position = self.positions[traj_idx, time_idx + 1]
        next_velocity = self.velocities[traj_idx, time_idx + 1]
        acceleration = (next_velocity - context.velocity) / self.config.dt

        context.y_pos = next_position
        context.y_velocity = next_velocity
        context.y_acceleration = acceleration
        return context, future
