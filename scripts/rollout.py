# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from particle_jepa.data.graph_builder import ParticleGraphBuilder
from particle_jepa.data.toy_dataset import ToyParticleConfig, generate_toy_rollouts
from particle_jepa.evaluation.rollout_eval import rollout_error, rollout_step
from particle_jepa.models import GraphNetworkSimulator, HybridGNSJEPA
from particle_jepa.utils.checkpointing import load_checkpoint
from particle_jepa.utils.perf import autocast_context, compile_model, strip_compiled_state_dict
from particle_jepa.visualization.animations import save_rollout_frame_strip


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a predicted-vs-ground-truth toy rollout.")
    parser.add_argument("--checkpoint", default=None, help="Path to a GNS or hybrid checkpoint.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
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
    model = _build_rollout_model(config).to(device)
    model.load_state_dict(strip_compiled_state_dict(checkpoint["model"]))
    model = compile_model(model, config)
    model.eval()

    gt, pred = _rollout(model, config, device, args.steps, args.seed)
    error = rollout_error(pred, gt).item()
    output = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent.parent / "visualizations" / "rollout_strip.png"
    )
    save_rollout_frame_strip(gt, pred, output=output)
    print(f"rollout position error: {error:.4f}")
    print(f"saved rollout strip: {output}")


def _rollout(model, config: dict, device: torch.device, steps: int, seed: int):
    data_cfg = config["data"]
    toy_config = ToyParticleConfig(
        num_trajectories=1,
        num_particles=int(data_cfg.get("num_particles", 64)),
        sequence_length=max(steps + 1, 2),
        dimension=int(data_cfg.get("dimension", 2)),
        dt=float(data_cfg.get("dt", 0.01)),
        future_offset=int(data_cfg.get("horizon", data_cfg.get("future_offset", 1))),
        radius=float(data_cfg.get("graph_radius", data_cfg.get("radius", 0.15))),
        seed=seed,
    )
    rollout = generate_toy_rollouts(toy_config)
    gt_positions = rollout["positions"][0, : steps + 1]
    velocities = rollout["velocities"][0, 0].to(device)
    positions = gt_positions[0].to(device)
    particle_types = rollout["particle_types"][0].to(device)
    graph_builder = ParticleGraphBuilder(radius=toy_config.radius)
    predicted = [positions.detach().cpu()]

    with torch.no_grad():
        for _ in range(steps):
            graph = graph_builder.build(positions, velocities, particle_type=particle_types).to(
                device
            )
            with autocast_context(device, config):
                outputs = model(graph)
                acceleration = outputs["acceleration"] if isinstance(outputs, dict) else outputs
            positions, velocities = rollout_step(
                positions.float(), velocities.float(), acceleration.float(), toy_config.dt
            )
            positions = positions.clamp(0.0, toy_config.box_size)
            predicted.append(positions.detach().cpu())

    return gt_positions.cpu(), torch.stack(predicted)


def _build_rollout_model(config: dict):
    model_cfg = config["model"]
    experiment = config.get("experiment", model_cfg.get("type", "gns"))
    kwargs = {
        "node_dim": model_cfg.get("node_dim", model_cfg.get("node_input_dim", 7)),
        "edge_dim": model_cfg.get("edge_dim", model_cfg.get("edge_input_dim", 6)),
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "message_passing_steps": model_cfg.get("message_passing_steps", 5),
        "dropout": model_cfg.get("dropout", 0.0),
        "mlp_layers": model_cfg.get("mlp_layers", 2),
    }
    if experiment == "hybrid":
        return HybridGNSJEPA(
            **kwargs,
            latent_dim=model_cfg.get("latent_dim", 128),
            max_horizon=model_cfg.get("max_horizon", 32),
        )
    if experiment == "gns":
        return GraphNetworkSimulator(**kwargs)
    msg = f"Rollout requires a GNS or hybrid checkpoint, got '{experiment}'."
    raise ValueError(msg)


def _latest_checkpoint() -> Path:
    checkpoints = sorted(Path("runs").glob("*_gns/checkpoints/last.pt")) + sorted(
        Path("runs").glob("*_hybrid/checkpoints/last.pt")
    )
    if not checkpoints:
        msg = "No GNS or hybrid checkpoint found under runs/*/checkpoints/last.pt."
        raise FileNotFoundError(msg)
    return checkpoints[-1]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
