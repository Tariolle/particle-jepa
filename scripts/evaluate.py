from __future__ import annotations

import argparse

import torch
from torch_geometric.loader import DataLoader

from particle_jepa.data.dataset import build_dataset
from particle_jepa.evaluation.metrics import cosine_alignment
from particle_jepa.models import ParticleJEPA
from particle_jepa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent alignment on toy graph pairs.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = build_dataset(config["data"])
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"])
    model = ParticleJEPA(**config["model"])
    model.eval()
    scores = []
    with torch.no_grad():
        for context, future in loader:
            outputs = model(context, future)
            scores.append(cosine_alignment(outputs["prediction"], outputs["target"]))
    score = torch.stack(scores).mean()
    print(f"latent cosine alignment: {score.item():.4f}")


if __name__ == "__main__":
    main()
