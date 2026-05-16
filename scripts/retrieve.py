# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from particle_jepa.data.dataset import build_dataset
from particle_jepa.evaluation.retrieval import (
    chance_retrieval_accuracy,
    nearest_future_indices,
    random_latent_retrieval_accuracy,
    retrieval_accuracy,
)
from particle_jepa.models import ParticleJEPA
from particle_jepa.utils.checkpointing import load_checkpoint
from particle_jepa.utils.perf import autocast_context, compile_model, strip_compiled_state_dict
from particle_jepa.visualization.particles import plot_retrieval_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Export latent future retrieval visualizations.")
    parser.add_argument("--checkpoint", default=None, help="Path to a Particle-JEPA checkpoint.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--random-trials", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    config.setdefault("train", {})
    config["train"]["precision"] = "fp16"
    config["train"]["compile"] = args.compile
    config["train"]["compile_mode"] = "reduce-overhead"

    device = _resolve_device(args.device)
    model = _build_jepa(config).to(device)
    model.load_state_dict(strip_compiled_state_dict(checkpoint["model"]))
    model = compile_model(model, config)
    model.eval()

    dataset = build_dataset(_small_data_config(config["data"], args.max_samples))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    predictions, targets, context_positions, future_positions = _collect_latents(
        model, loader, device, config
    )
    top_k = min(args.top_k, targets.size(0))
    indices = nearest_future_indices(predictions, targets, top_k=top_k)
    metrics = {
        "top1_accuracy": retrieval_accuracy(predictions, targets, top_k=1).item(),
        f"top{top_k}_accuracy": retrieval_accuracy(predictions, targets, top_k=top_k).item(),
        "chance_top1_accuracy": chance_retrieval_accuracy(targets.size(0), top_k=1),
        f"chance_top{top_k}_accuracy": chance_retrieval_accuracy(targets.size(0), top_k=top_k),
        "random_top1_accuracy": random_latent_retrieval_accuracy(
            targets, top_k=1, trials=args.random_trials
        ).item(),
        f"random_top{top_k}_accuracy": random_latent_retrieval_accuracy(
            targets, top_k=top_k, trials=args.random_trials
        ).item(),
        "num_samples": int(targets.size(0)),
        "prediction_target_cosine": F.cosine_similarity(predictions, targets, dim=-1).mean().item(),
        "prediction_latent_std": predictions.std(dim=0).mean().item(),
        "target_latent_std": targets.std(dim=0).mean().item(),
    }
    metrics["top1_lift_vs_chance"] = metrics["top1_accuracy"] / max(
        metrics["chance_top1_accuracy"], 1e-12
    )
    metrics[f"top{top_k}_lift_vs_chance"] = metrics[f"top{top_k}_accuracy"] / max(
        metrics[f"chance_top{top_k}_accuracy"], 1e-12
    )

    query = min(max(args.query_index, 0), targets.size(0) - 1)
    retrieved_positions = [future_positions[idx] for idx in indices[query].tolist()]
    output = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent.parent / "visualizations" / "retrieval_panel.png"
    )
    plot_retrieval_panel(
        context_positions[query],
        future_positions[query],
        retrieved_positions,
        output=output,
    )
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved retrieval panel: {output}")


def _collect_latents(model, loader, device: torch.device, config: dict):
    predictions = []
    targets = []
    context_positions = []
    future_positions = []
    with torch.no_grad():
        for context, future in loader:
            context = context.to(device)
            future = future.to(device)
            with autocast_context(device, config):
                outputs = model(context, future)
            predictions.append(F.normalize(outputs["prediction"].float().cpu(), dim=-1))
            targets.append(F.normalize(outputs["target"].float().cpu(), dim=-1))
            context_positions.extend(
                context.pos.detach().cpu().split(_nodes_per_graph(context), dim=0)
            )
            future_positions.extend(
                future.pos.detach().cpu().split(_nodes_per_graph(future), dim=0)
            )
    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        context_positions,
        future_positions,
    )


def _nodes_per_graph(batch) -> list[int]:
    counts = torch.bincount(batch.batch.detach().cpu())
    return counts.tolist()


def _build_jepa(config: dict) -> ParticleJEPA:
    model_cfg = config["model"]
    return ParticleJEPA(
        node_dim=model_cfg.get("node_dim", model_cfg.get("node_input_dim", 7)),
        edge_dim=model_cfg.get("edge_dim", model_cfg.get("edge_input_dim", 6)),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        latent_dim=model_cfg.get("latent_dim", 128),
        message_passing_steps=model_cfg.get("message_passing_steps", 5),
        dropout=model_cfg.get("dropout", 0.0),
        mlp_layers=model_cfg.get("mlp_layers", 2),
        max_horizon=model_cfg.get("max_horizon", 32),
    )


def _small_data_config(data_config: dict, max_samples: int) -> dict:
    config = dict(data_config)
    sequence_length = int(config.get("trajectory_length", config.get("sequence_length", 100)))
    horizon = int(config.get("horizon", config.get("future_offset", 5)))
    samples_per_trajectory = max(sequence_length - horizon, 1)
    trajectories = max(
        1,
        min(
            int(config.get("num_train_trajectories", 16)), max_samples // samples_per_trajectory + 1
        ),
    )
    config["num_train_trajectories"] = trajectories
    config["num_val_trajectories"] = 0
    return config


def _latest_checkpoint() -> Path:
    checkpoints = sorted(Path("runs").glob("*_jepa/checkpoints/last.pt"))
    if not checkpoints:
        msg = "No Particle-JEPA checkpoint found under runs/*_jepa/checkpoints/last.pt."
        raise FileNotFoundError(msg)
    return checkpoints[-1]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
