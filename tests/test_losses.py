from __future__ import annotations

import torch
from torch import nn

from particle_jepa.training.losses import acceleration_loss, jepa_loss


def test_acceleration_loss_is_zero_for_equal_tensors() -> None:
    target = torch.ones(4, 2)
    assert acceleration_loss(target, target).item() == 0.0


def test_jepa_loss_is_finite() -> None:
    prediction = torch.randn(3, 8)
    target = torch.randn(3, 8)
    loss = jepa_loss(prediction, target)
    assert torch.isfinite(loss)


def test_jepa_loss_decreases_on_tiny_overfit_batch() -> None:
    torch.manual_seed(0)
    predictor = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.05)
    context = torch.randn(8, 4)
    target = torch.randn(8, 4)

    first_loss = None
    last_loss = None
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = jepa_loss(predictor(context), target)
        if step == 0:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

    assert first_loss is not None
    assert last_loss is not None
    assert last_loss < first_loss
