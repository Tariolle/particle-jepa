from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def rmse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(F.mse_loss(prediction, target))


def mean_position_error(predicted_positions: Tensor, target_positions: Tensor) -> Tensor:
    return torch.linalg.norm(predicted_positions - target_positions, dim=-1).mean()


def one_step_position_mse(predicted_positions: Tensor, target_positions: Tensor) -> Tensor:
    return F.mse_loss(predicted_positions, target_positions)


def one_step_velocity_mse(predicted_velocities: Tensor, target_velocities: Tensor) -> Tensor:
    return F.mse_loss(predicted_velocities, target_velocities)


def rollout_position_mse(predicted_rollout: Tensor, target_rollout: Tensor) -> Tensor:
    return F.mse_loss(predicted_rollout, target_rollout)


def chamfer_distance(a: Tensor, b: Tensor) -> Tensor:
    pairwise = torch.cdist(a, b).pow(2)
    return pairwise.min(dim=-1).values.mean() + pairwise.min(dim=-2).values.mean()


def cosine_alignment(a: Tensor, b: Tensor) -> Tensor:
    return F.cosine_similarity(a, b, dim=-1).mean()


def latent_prediction_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(prediction, target)


def latent_trajectory_alignment(predicted: Tensor, target: Tensor) -> Tensor:
    pred_delta = predicted[1:] - predicted[:-1]
    target_delta = target[1:] - target[:-1]
    return cosine_alignment(pred_delta, target_delta)
