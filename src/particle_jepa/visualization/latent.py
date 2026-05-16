from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_latent_trajectory(latents: Tensor, output: str | Path | None = None):
    latents = torch.as_tensor(latents).detach().cpu()
    if latents.size(-1) < 2:
        msg = "Need at least two latent dimensions to plot a trajectory."
        raise ValueError(msg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(latents[:, 0], latents[:, 1], marker="o", linewidth=1.5)
    ax.set_xlabel("latent 0")
    ax.set_ylabel("latent 1")
    ax.set_title("Latent trajectory")
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig, ax
