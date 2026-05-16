from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_particles(positions: Tensor, output: str | Path | None = None, title: str = "Particles"):
    positions = torch.as_tensor(positions).detach().cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(positions[:, 0], positions[:, 1], s=28, c=positions[:, 0], cmap="viridis")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig, ax


def plot_retrieval_panel(
    current: Tensor,
    true_future: Tensor,
    retrieved: list[Tensor],
    output: str | Path | None = None,
):
    """Show current, true future, and nearest retrieved future particle states."""
    frames = [current, true_future, *retrieved[:3]]
    titles = ["current", "true future", "retrieved 1", "retrieved 2", "retrieved 3"][: len(frames)]
    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3), sharex=True, sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for ax, frame, title in zip(axes, frames, titles, strict=True):
        frame = torch.as_tensor(frame).detach().cpu()
        ax.scatter(frame[:, 0], frame[:, 1], s=18, c=frame[:, 0], cmap="viridis")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig, axes
