# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from particle_jepa.data.dataset import build_dataset
from particle_jepa.training.train_gns import train_gns
from particle_jepa.training.train_jepa import train_jepa
from particle_jepa.utils.checkpointing import save_checkpoint
from particle_jepa.utils.config import load_config, normalize_experiment_config
from particle_jepa.utils.perf import unwrap_compiled_model
from particle_jepa.utils.runs import copy_config, create_run_dir
from particle_jepa.utils.seed import seed_everything
from particle_jepa.utils.tracking import init_tracker


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    if "--config" in sys.argv:
        legacy_main()
    else:
        hydra_main()


@hydra.main(version_base="1.3", config_path="../configs/model", config_name="particle_jepa")
def hydra_main(cfg: DictConfig) -> None:
    config = normalize_experiment_config(OmegaConf.to_container(cfg, resolve=True))
    run_training(config, config_source=None)


def legacy_main() -> None:
    parser = argparse.ArgumentParser(description="Train Particle-JEPA experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", choices=["gns", "jepa"], default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = normalize_experiment_config(load_config(args.config))
    experiment = args.experiment or _canonical_experiment(config.get("experiment", "jepa"))
    run_training(
        config, experiment=experiment, config_source=args.config, checkpoint=args.checkpoint
    )


def run_training(
    config: dict,
    experiment: str | None = None,
    config_source: str | None = None,
    checkpoint: str | None = None,
) -> None:
    experiment = experiment or _canonical_experiment(config.get("experiment", "jepa"))
    seed_everything(config.get("seed", 7))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["data"])
    train_dataset, val_dataset = _split_dataset(dataset, config["data"], config.get("seed", 7))
    run_root = config.get("paths", {}).get("run_root", "runs")
    run_dir = create_run_dir(experiment, run_root)
    if config_source is not None:
        copy_config(config_source, run_dir)
    else:
        (run_dir / "config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create(config.get("raw_config", config))),
            encoding="utf-8",
        )
    tracker = init_tracker(config, run_dir)

    try:
        if experiment == "gns":
            model = train_gns(
                train_dataset,
                config,
                device,
                val_dataset=val_dataset,
                run_dir=run_dir,
                tracker=tracker,
            )
        elif experiment == "jepa":
            model = train_jepa(
                train_dataset,
                config,
                device,
                val_dataset=val_dataset,
                run_dir=run_dir,
                tracker=tracker,
            )
        else:
            msg = f"Unsupported experiment '{experiment}'. Expected 'jepa' or 'gns'."
            raise ValueError(msg)
    finally:
        tracker.finish()

    checkpoint_path = checkpoint
    if checkpoint_path is None:
        checkpoint_path = run_dir / "checkpoints" / "last.pt"
    save_checkpoint(
        {"model": unwrap_compiled_model(model).state_dict(), "config": config}, checkpoint_path
    )
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"checkpoint": str(checkpoint_path)}, indent=2), encoding="utf-8"
    )
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"run directory: {run_dir}")


def _canonical_experiment(value: str) -> str:
    aliases = {"particle_jepa": "jepa"}
    return aliases.get(value, value)


def _split_dataset(dataset, data_config: dict, seed: int):
    val_trajectories = int(data_config.get("num_val_trajectories", 0))
    total_trajectories = int(data_config.get("num_trajectories", 0)) or (
        int(data_config.get("num_train_trajectories", 0)) + val_trajectories
    )
    if val_trajectories <= 0 or total_trajectories <= 0:
        return dataset, None
    val_fraction = min(max(val_trajectories / total_trajectories, 0.0), 0.5)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


if __name__ == "__main__":
    main()
