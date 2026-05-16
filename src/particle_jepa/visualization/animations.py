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
