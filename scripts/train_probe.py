# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from particle_jepa.data.dataset import build_dataset
from particle_jepa.data.lts_dataset import LearningToSimulateConfig, LearningToSimulateDataset
from particle_jepa.evaluation.rollout_eval import rollout_error, rollout_step
from particle_jepa.models import ParticleJEPA
from particle_jepa.models.decoders import AccelerationDecoder
from particle_jepa.training.losses import acceleration_loss
from particle_jepa.utils.checkpointing import load_checkpoint, save_checkpoint
from particle_jepa.utils.perf import (
    autocast_context,
    compile_model,
    configure_inductor,
    make_grad_scaler,
    make_pyg_dataloader,
    move_to_device,
    strip_compiled_state_dict,
    unwrap_compiled_model,
)
from particle_jepa.utils.runs import append_jsonl, create_run_dir
from particle_jepa.visualization.animations import (
    animate_rollout_comparison,
    save_rollout_frame_strip,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an evaluation-only acceleration probe on frozen Particle-JEPA latents."
    )
    parser.add_argument("--checkpoint", default=None, help="Path to a Particle-JEPA checkpoint.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument(
        "--latent-source",
        choices=["node_prediction", "node_context", "node_target", "raw_features"],
        default="node_prediction",
    )
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--rollout-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _latest_jepa_checkpoint()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    config.setdefault("train", {})
    config["train"]["precision"] = "fp16"
    config["train"]["compile"] = args.compile
    config["train"]["compile_mode"] = "reduce-overhead"
    _force_one_step_probe_data(config)
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
        config["data"]["batch_size"] = args.batch_size

    device = _resolve_device(args.device)
    run_dir = create_run_dir("jepa_probe", config.get("paths", {}).get("run_root", "runs"))
    probe_config = {
        "jepa_checkpoint": str(checkpoint_path),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "mlp_layers": args.mlp_layers,
        "latent_source": args.latent_source,
        "rollout_steps": args.rollout_steps,
        "rollout_start": args.rollout_start,
        "base_config": config,
    }
    (run_dir / "config.json").write_text(json.dumps(probe_config, indent=2), encoding="utf-8")

    jepa = _build_jepa(config).to(device)
    jepa.load_state_dict(strip_compiled_state_dict(checkpoint["model"]))
    jepa.eval()
    for parameter in jepa.parameters():
        parameter.requires_grad_(False)
    predict_latents = _compile_predict(jepa.predict, config)

    decoder = AccelerationDecoder(
        _probe_input_dim(config, args.latent_source),
        hidden_dim=args.hidden_dim,
        spatial_dim=_spatial_dim(config),
        mlp_layers=args.mlp_layers,
    ).to(device)
    decoder_config = _decoder_compile_config(config)
    decoder = compile_model(decoder, decoder_config)

    dataset = build_dataset(config["data"])
    loader = make_pyg_dataloader(dataset, config, device, shuffle=True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, config)

    for epoch in range(args.epochs):
        decoder.train()
        running = 0.0
        skipped = 0
        seen = 0
        for context, future in tqdm(loader, desc=f"probe epoch {epoch + 1}", leave=False):
            seen += 1
            try:
                context = move_to_device(context, device)
                future = move_to_device(future, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), autocast_context(device, config):
                    latents = _probe_latents(
                        jepa,
                        predict_latents,
                        context,
                        future,
                        args.latent_source,
                        device,
                    )
                with autocast_context(device, config):
                    acceleration = decoder(latents.detach())
                    loss = acceleration_loss(
                        acceleration.float(),
                        context.y_acceleration.float(),
                        getattr(context, "dynamic_mask", None),
                    )
            except torch.OutOfMemoryError:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        used = max(seen - skipped, 1)
        row = {
            "epoch": epoch + 1,
            "latent_source": args.latent_source,
            "train_loss": running / used,
            "skipped_batches": skipped,
        }
        append_jsonl(run_dir / "logs.jsonl", row)
        print(f"epoch={epoch + 1} probe_loss={row['train_loss']:.6f} skipped={skipped}")

    checkpoint_out = run_dir / "checkpoints" / "last.pt"
    save_checkpoint(
        {
            "decoder": unwrap_compiled_model(decoder).state_dict(),
            "jepa_checkpoint": str(checkpoint_path),
            "config": config,
            "probe_config": probe_config,
        },
        checkpoint_out,
    )
    gt, pred, bounds = rollout_probe(
        jepa,
        decoder,
        config,
        device,
        latent_source=args.latent_source,
        predict_latents=predict_latents,
        steps=args.rollout_steps,
        start=args.rollout_start,
    )
    error = rollout_error(pred, gt).item()
    gif_path = run_dir / "visualizations" / "jepa_probe_rollout.gif"
    strip_path = run_dir / "visualizations" / "jepa_probe_rollout_strip.png"
    animate_rollout_comparison(gt, pred, output=gif_path, bounds=bounds)
    save_rollout_frame_strip(gt, pred, output=strip_path, bounds=bounds)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_out),
                "rollout_position_mse": error,
                "rollout_gif": str(gif_path),
                "rollout_strip": str(strip_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"rollout position error: {error:.6f}")
    print(f"saved checkpoint: {checkpoint_out}")
    print(f"saved rollout gif: {gif_path}")
    print(f"saved rollout strip: {strip_path}")
    print(f"run directory: {run_dir}")


def rollout_probe(
    jepa,
    decoder,
    config: dict,
    device: torch.device,
    latent_source: str,
    predict_latents,
    steps: int,
    start: int,
):
    data_cfg = config["data"]
    if data_cfg.get("dataset") not in {"learning_to_simulate", "lts"}:
        msg = "The probe rollout currently targets Learning-to-Simulate datasets."
        raise ValueError(msg)
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
    particle_types = trajectory["particle_types"].to(device)
    kinematic_id = int(data_cfg.get("kinematic_particle_id", 3))
    kinematic_mask = particle_types == kinematic_id
    predicted = [current_sequence[-1].detach().cpu()]
    bounds = dataset.metadata.get("bounds")

    jepa.eval()
    decoder.eval()
    with torch.no_grad():
        for _ in range(steps):
            graph = dataset.build_graph_from_position_sequence(
                current_sequence.detach().cpu(), trajectory["particle_types"]
            ).to(device)
            with autocast_context(device, config):
                latents = _probe_latents(
                    jepa,
                    predict_latents,
                    graph,
                    graph,
                    latent_source,
                    device,
                )
                acceleration = decoder(latents).float()
            acceleration = _denormalize_lts_acceleration(acceleration, dataset, data_cfg)
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


def _build_jepa(config: dict) -> ParticleJEPA:
    model_cfg = config["model"]
    return ParticleJEPA(
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        message_passing_steps=model_cfg["message_passing_steps"],
        dropout=model_cfg.get("dropout", 0.0),
        mlp_layers=model_cfg.get("mlp_layers", 2),
        max_horizon=model_cfg.get("max_horizon", 32),
        latent_predictor_steps=model_cfg.get("latent_predictor_steps", 2),
        region_grid_size=model_cfg.get("region_grid_size", 4),
        predictor_type=model_cfg.get("predictor_type", "message_passing"),
        predictor_layers=model_cfg.get("predictor_layers"),
        predictor_heads=model_cfg.get("predictor_heads", 4),
        predictor_dropout=model_cfg.get("predictor_dropout"),
    )


def _compile_predict(predict_fn, config: dict):
    train_cfg = config.get("train", {})
    if not train_cfg.get("compile", True):
        return predict_fn
    configure_inductor(config)
    kwargs = {}
    if train_cfg.get("compile_dynamic", None) is not None:
        kwargs["dynamic"] = bool(train_cfg["compile_dynamic"])
    try:
        return torch.compile(
            predict_fn,
            mode=train_cfg.get("compile_mode", "reduce-overhead"),
            fullgraph=train_cfg.get("compile_fullgraph", False),
            **kwargs,
        )
    except Exception as exc:
        print(f"torch.compile unavailable for JEPA predict path, continuing eager: {exc}")
        return predict_fn


def _probe_latents(
    jepa,
    predict_latents,
    context,
    future,
    latent_source: str,
    device: torch.device,
) -> torch.Tensor:
    if latent_source == "raw_features":
        return context.x
    horizon = torch.ones(
        int(getattr(context, "num_graphs", 1)),
        dtype=torch.long,
        device=device,
    )
    if latent_source == "node_target":
        return jepa(context, future, horizon=horizon)["node_target"]
    outputs = predict_latents(context, horizon=horizon)
    return outputs[latent_source]


def _probe_input_dim(config: dict, latent_source: str) -> int:
    if latent_source != "raw_features":
        return int(config["model"]["latent_dim"])
    return int(config["model"]["node_dim"])


def _force_one_step_probe_data(config: dict) -> None:
    data_cfg = config.setdefault("data", {})
    data_cfg["horizon"] = 1
    data_cfg["future_offset"] = 1
    data_cfg.pop("horizons", None)
    data_cfg.pop("future_offsets", None)


def _decoder_compile_config(config: dict) -> dict:
    decoder_config = dict(config)
    decoder_config["train"] = dict(config.get("train", {}))
    decoder_config["train"]["compile_scope"] = "full"
    return decoder_config


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


def _spatial_dim(config: dict) -> int:
    metadata_path = (
        Path(config["data"].get("data_root", "data/raw/WaterDropSample")) / "metadata.json"
    )
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return int(metadata.get("dim", 2))
    return 2


def _latest_jepa_checkpoint() -> Path:
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
