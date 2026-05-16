from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from torch import Tensor


def animate_rollout(positions: Tensor, output: str | Path | None = None, interval: int = 80):
    positions = torch.as_tensor(positions).detach().cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter([], [], s=25)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    def update(frame: int):
        scatter.set_offsets(positions[frame])
        ax.set_title(f"t={frame}")
        return (scatter,)

    animation = FuncAnimation(fig, update, frames=positions.size(0), interval=interval, blit=True)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output)
    return animation


def animate_rollout_comparison(
    ground_truth: Tensor,
    predicted: Tensor,
    output: str | Path | None = None,
    interval: int = 80,
):
    """Animate ground-truth and predicted rollouts side by side with an overlay panel."""
    ground_truth = torch.as_tensor(ground_truth).detach().cpu()
    predicted = torch.as_tensor(predicted).detach().cpu()
    frames = min(ground_truth.size(0), predicted.size(0))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    scatters = []
    for ax, title in zip(axes, ["ground truth", "predicted", "overlay"], strict=True):
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
    scatters.append(axes[0].scatter([], [], s=20))
    scatters.append(axes[1].scatter([], [], s=20))
    scatters.append((axes[2].scatter([], [], s=16), axes[2].scatter([], [], s=16, alpha=0.6)))

    def update(frame: int):
        gt = ground_truth[frame]
        pred = predicted[frame]
        scatters[0].set_offsets(gt)
        scatters[1].set_offsets(pred)
        scatters[2][0].set_offsets(gt)
        scatters[2][1].set_offsets(pred)
        fig.suptitle(f"t={frame}")
        return (scatters[0], scatters[1], scatters[2][0], scatters[2][1])

    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output)
    return animation
