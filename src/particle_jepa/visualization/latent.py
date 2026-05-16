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


def plot_latent_trajectory_comparison(
    predicted: Tensor,
    target: Tensor,
    output: str | Path | None = None,
):
    """Plot predicted and target latent trajectories using their first two coordinates."""
    predicted = torch.as_tensor(predicted).detach().cpu()
    target = torch.as_tensor(target).detach().cpu()
    if predicted.size(-1) < 2 or target.size(-1) < 2:
        msg = "Need at least two latent dimensions to plot trajectories."
        raise ValueError(msg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(target[:, 0], target[:, 1], marker="o", linewidth=1.5, label="target")
    ax.plot(predicted[:, 0], predicted[:, 1], marker="x", linewidth=1.5, label="predicted")
    ax.set_xlabel("latent 0")
    ax.set_ylabel("latent 1")
    ax.set_title("Latent trajectory alignment")
    ax.legend()
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig, ax
