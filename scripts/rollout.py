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

from particle_jepa.data.lts_dataset import LearningToSimulateConfig, LearningToSimulateDataset
from particle_jepa.data.graph_builder import ParticleGraphBuilder
from particle_jepa.data.toy_dataset import ToyParticleConfig, generate_toy_rollouts
from particle_jepa.evaluation.rollout_eval import rollout_error, rollout_step
from particle_jepa.models import GraphNetworkSimulator
from particle_jepa.utils.checkpointing import load_checkpoint
from particle_jepa.utils.perf import autocast_context, compile_model, strip_compiled_state_dict
from particle_jepa.visualization.animations import (
    animate_rollout_comparison,
    save_rollout_frame_strip,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a predicted-vs-ground-truth toy rollout.")
    parser.add_argument("--checkpoint", default=None, help="Path to a GNS checkpoint.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
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

    gt, pred, bounds = _rollout(model, config, device, args.steps, args.seed, args.start)
    error = rollout_error(pred, gt).item()
    output = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent.parent / "visualizations" / "rollout_strip.png"
    )
    if output.suffix.lower() == ".gif":
        animate_rollout_comparison(gt, pred, output=output, bounds=bounds)
    else:
        save_rollout_frame_strip(gt, pred, output=output, bounds=bounds)
    print(f"rollout position error: {error:.4f}")
    print(f"saved rollout visualization: {output}")


def _rollout(model, config: dict, device: torch.device, steps: int, seed: int, start: int):
    data_cfg = config["data"]
    if data_cfg.get("dataset") in {"learning_to_simulate", "lts"}:
        return _rollout_lts(model, data_cfg, config, device, steps, start)
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

    return gt_positions.cpu(), torch.stack(predicted), None


def _rollout_lts(model, data_cfg: dict, config: dict, device: torch.device, steps: int, start: int):
    lts_config = LearningToSimulateConfig(
        root=data_cfg.get("data_root", data_cfg.get("root", "data/raw/WaterDropSample")),
        split=data_cfg.get("split", "train"),
        future_offset=int(data_cfg.get("horizon", data_cfg.get("future_offset", 1))),
        radius=data_cfg.get("graph_radius", data_cfg.get("radius")),
        max_neighbors=data_cfg.get("max_neighbors"),
        max_trajectories=1,
        sample_stride=1,
        max_samples_per_trajectory=1,
        normalize_acceleration=bool(data_cfg.get("normalize_acceleration", True)),
        kinematic_particle_id=int(data_cfg.get("kinematic_particle_id", 3)),
        input_sequence_length=int(data_cfg.get("input_sequence_length", 6)),
        num_particle_types=int(data_cfg.get("num_particle_types", 9)),
        noise_std=float(data_cfg.get("noise_std", 0.0)),
        apply_noise=False,
        use_official_features=bool(data_cfg.get("use_official_features", True)),
    )
    dataset = LearningToSimulateDataset(lts_config)
    trajectory = dataset.trajectories[0]
    history = int(lts_config.input_sequence_length)
    start = min(max(start, history - 1), trajectory["positions"].size(0) - steps - 1)
    gt_positions = trajectory["positions"][start : start + steps + 1]
    gt_velocities = trajectory["velocities"][start : start + steps + 1]
    current_sequence = trajectory["positions"][start - history + 1 : start + 1].to(device)
    positions = current_sequence[-1]
    particle_types = trajectory["particle_types"].to(device)
    kinematic_id = int(data_cfg.get("kinematic_particle_id", 3))
    kinematic_mask = particle_types == kinematic_id
    predicted = [positions.detach().cpu()]
    bounds = dataset.metadata.get("bounds")

    with torch.no_grad():
        for _ in range(steps):
            graph = dataset.build_graph_from_position_sequence(
                current_sequence.detach().cpu(), trajectory["particle_types"]
            ).to(device)
            with autocast_context(device, config):
                outputs = model(graph)
                acceleration = outputs["acceleration"] if isinstance(outputs, dict) else outputs
            acceleration = _denormalize_lts_acceleration(acceleration.float(), dataset, data_cfg)
            velocities = current_sequence[-1] - current_sequence[-2]
            positions, velocities = rollout_step(
                current_sequence[-1].float(), velocities.float(), acceleration.float(), 1.0
            )
            if torch.any(kinematic_mask):
                positions = torch.where(
                    kinematic_mask[:, None], gt_positions[len(predicted)].to(device), positions
                )
                velocities = torch.where(
                    kinematic_mask[:, None], gt_velocities[len(predicted)].to(device), velocities
                )
            if bounds is not None:
                bounds_tensor = torch.tensor(bounds, dtype=positions.dtype, device=positions.device)
                positions = torch.maximum(positions, bounds_tensor[:, 0])
                positions = torch.minimum(positions, bounds_tensor[:, 1])
            current_sequence = torch.cat([current_sequence[1:], positions[None]], dim=0)
            predicted.append(positions.detach().cpu())

    return gt_positions.cpu(), torch.stack(predicted), bounds


def _denormalize_lts_acceleration(acceleration, dataset, data_cfg: dict):
    if not bool(data_cfg.get("normalize_acceleration", True)):
        return acceleration
    mean = torch.tensor(
        dataset.metadata["acc_mean"], dtype=acceleration.dtype, device=acceleration.device
    )
    std = torch.tensor(
        dataset.metadata["acc_std"], dtype=acceleration.dtype, device=acceleration.device
    ).clamp_min(1e-8)
    noise_std = float(data_cfg.get("noise_std", 0.0))
    if noise_std > 0.0:
        std = torch.sqrt(std.pow(2) + noise_std**2)
    return acceleration * std + mean


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
    if experiment == "gns":
        return GraphNetworkSimulator(**kwargs)
    msg = f"Rollout requires a GNS checkpoint, got '{experiment}'."
    raise ValueError(msg)


def _latest_checkpoint() -> Path:
    checkpoints = sorted(Path("runs").glob("*_gns/checkpoints/last.pt"))
    if not checkpoints:
        msg = "No GNS checkpoint found under runs/*_gns/checkpoints/last.pt."
        raise FileNotFoundError(msg)
    return checkpoints[-1]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
