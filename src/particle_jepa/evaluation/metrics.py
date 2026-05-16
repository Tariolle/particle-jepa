from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def rmse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(F.mse_loss(prediction, target))


def mean_position_error(predicted_positions: Tensor, target_positions: Tensor) -> Tensor:
    return torch.linalg.norm(predicted_positions - target_positions, dim=-1).mean()


def cosine_alignment(a: Tensor, b: Tensor) -> Tensor:
    return F.cosine_similarity(a, b, dim=-1).mean()
