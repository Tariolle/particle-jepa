from __future__ import annotations

import torch

from particle_jepa.training.losses import acceleration_loss, jepa_loss


def test_acceleration_loss_is_zero_for_equal_tensors() -> None:
    target = torch.ones(4, 2)
    assert acceleration_loss(target, target).item() == 0.0


def test_jepa_loss_is_finite() -> None:
    prediction = torch.randn(3, 8)
    target = torch.randn(3, 8)
    loss = jepa_loss(prediction, target)
    assert torch.isfinite(loss)
