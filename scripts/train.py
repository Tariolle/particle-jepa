from __future__ import annotations

import argparse
from pathlib import Path

import torch

from particle_jepa.data.dataset import build_dataset
from particle_jepa.training.train_gns import train_gns
from particle_jepa.training.train_hybrid import train_hybrid
from particle_jepa.training.train_jepa import train_jepa
from particle_jepa.utils.checkpointing import save_checkpoint
from particle_jepa.utils.config import load_config
from particle_jepa.utils.seed import seed_everything


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Particle-JEPA experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", choices=["gns", "jepa", "hybrid"], default="jepa")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 7))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["data"])

    if args.experiment == "gns":
        model = train_gns(dataset, config, device)
    elif args.experiment == "hybrid":
        model = train_hybrid(dataset, config, device)
    else:
        model = train_jepa(dataset, config, device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / f"{args.experiment}.pt"
    save_checkpoint({"model": model.state_dict(), "config": config}, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
