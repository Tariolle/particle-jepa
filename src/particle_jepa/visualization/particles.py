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
