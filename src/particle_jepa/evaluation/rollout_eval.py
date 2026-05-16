from __future__ import annotations

from torch import Tensor

from particle_jepa.evaluation.metrics import mean_position_error


def rollout_step(
    positions: Tensor, velocities: Tensor, acceleration: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    next_velocity = velocities + dt * acceleration
    next_position = positions + dt * next_velocity
    return next_position, next_velocity


def rollout_error(predicted: Tensor, target: Tensor) -> Tensor:
    return mean_position_error(predicted, target)
