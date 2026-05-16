from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a config dictionary."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy and standalone config schemas into one training shape."""
    if "project" in config:
        model = config.get("model", {})
        data = config.get("data", {})
        optimizer = config.get("optimizer", {})
        training = config.get("training", {})
        loss = config.get("loss", {})
        return {
            "seed": config.get("project", {}).get("seed", 7),
            "device": training.get("device", "auto"),
            "experiment": model.get("type", "particle_jepa"),
            "project": config.get("project", {}),
            "data": data,
            "model": {
                "node_dim": model.get("node_input_dim", model.get("node_dim", 7)),
                "edge_dim": model.get("edge_input_dim", model.get("edge_dim", 6)),
                "hidden_dim": model.get("hidden_dim", 128),
                "latent_dim": model.get("latent_dim", model.get("hidden_dim", 128)),
                "message_passing_steps": model.get("message_passing_steps", 5),
                "mlp_layers": model.get("mlp_layers", 2),
                "dropout": model.get("dropout", 0.0),
                "max_horizon": model.get("max_horizon", 32),
                "latent_predictor_steps": model.get("latent_predictor_steps", 2),
            },
            "train": {
                "batch_size": data.get("batch_size", 8),
                "epochs": training.get("epochs", 10),
                "learning_rate": optimizer.get("lr", 3e-4),
                "weight_decay": optimizer.get("weight_decay", 1e-4),
                "grad_clip_norm": training.get("grad_clip_norm", 1.0),
                "checkpoint_every": training.get("checkpoint_every", 5),
                "visualize_every": training.get("visualize_every", 5),
                "precision": training.get("precision", "fp16"),
                "compile": training.get("compile", True),
                "compile_mode": training.get("compile_mode", "reduce-overhead"),
                "compile_fullgraph": training.get("compile_fullgraph", False),
                "dynamics_loss_weight": loss.get("dynamics_loss_weight", 1.0),
                "jepa_loss_weight": loss.get("jepa_loss_weight", 0.2),
                "prediction_weight": loss.get("prediction_weight", 1.0),
                "node_prediction_weight": loss.get("node_prediction_weight", 1.0),
                "sigreg_weight": loss.get("sigreg_weight", 0.05),
                "sigreg_sketch_dim": loss.get("sigreg_sketch_dim", 64),
            },
            "paths": config.get("paths", {"run_root": "runs"}),
            "tracking": config.get("tracking", {"provider": "wandb", "enabled": False}),
            "raw_config": config,
        }
    normalized = dict(config)
    normalized.setdefault("experiment", "particle_jepa")
    normalized.setdefault("tracking", {"provider": "wandb", "enabled": False})
    return normalized
