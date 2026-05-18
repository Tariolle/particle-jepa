from __future__ import annotations

import time

import torch
from tqdm import tqdm

from particle_jepa.models import ParticleJEPA
from particle_jepa.training.losses import temporal_graph_jepa_loss
from particle_jepa.utils.checkpointing import save_checkpoint
from particle_jepa.utils.perf import (
    autocast_context,
    compile_model,
    make_grad_scaler,
    make_pyg_dataloader,
    move_to_device,
    unwrap_compiled_model,
)
from particle_jepa.utils.runs import append_jsonl


def train_jepa(
    dataset, config: dict, device: torch.device, val_dataset=None, run_dir=None, tracker=None
) -> ParticleJEPA:
    model_cfg = config["model"]
    train_cfg = config["train"]
    model = ParticleJEPA(
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
    ).to(device)
    model = compile_model(model, config)
    loader = make_pyg_dataloader(dataset, config, device, shuffle=True)
    val_loader = (
        make_pyg_dataloader(val_dataset, config, device, shuffle=False)
        if val_dataset is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scaler = make_grad_scaler(device, config)
    _run_compile_preflight(model, loader, optimizer, scaler, config, device, run_dir)

    for epoch in range(train_cfg["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        running_parts: dict[str, float] = {}
        skipped_batches = 0
        seen_batches = 0
        for context, future in tqdm(loader, desc=f"jepa epoch {epoch + 1}", leave=False):
            batch_start = time.perf_counter()
            seen_batches += 1
            try:
                context = move_to_device(context, device)
                future = move_to_device(future, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, config):
                    outputs = model(context, future)
                    losses = temporal_graph_jepa_loss(outputs, config)
                    loss = losses["loss"]
            except torch.OutOfMemoryError:
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            if not torch.isfinite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.get("grad_clip_norm", 1.0)
            )
            if not torch.isfinite(grad_norm):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            for name, value in losses.items():
                if name == "loss":
                    continue
                running_parts[name] = running_parts.get(name, 0.0) + value.item()
            _raise_if_batch_too_slow(time.perf_counter() - batch_start, train_cfg)
        used_batches = max(seen_batches - skipped_batches, 1)
        train_loss = running / used_batches
        val_loss = _evaluate(model, val_loader, device) if val_loader is not None else None
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "skipped_batches": skipped_batches,
            "epoch_seconds": epoch_seconds,
            "batches_per_second": seen_batches / max(epoch_seconds, 1e-9),
            **{name: value / used_batches for name, value in running_parts.items()},
        }
        if run_dir is not None:
            append_jsonl(run_dir / "logs.jsonl", row)
        if tracker is not None:
            tracker.log(row, step=epoch + 1)
        print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={val_loss}")
        _raise_if_epoch_too_slow(epoch_seconds, train_cfg)
        if run_dir is not None and (epoch + 1) % train_cfg.get("checkpoint_every", 1) == 0:
            save_checkpoint(
                {
                    "model": unwrap_compiled_model(model).state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                run_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
            )
    return model


def _run_compile_preflight(
    model: ParticleJEPA,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    device: torch.device,
    run_dir,
) -> None:
    train_cfg = config["train"]
    if not train_cfg.get("compile", True):
        if train_cfg.get("compile_required", True):
            msg = "compile_required=true but training.compile=false."
            raise RuntimeError(msg)
        return
    batches = int(train_cfg.get("preflight_batches", 0))
    warmup = int(train_cfg.get("preflight_warmup_batches", 0))
    if batches <= 0 and warmup <= 0:
        return

    model.train()
    measured_batches = 0
    measured_examples = 0
    skipped_batches = 0
    peak_memory_mb = 0.0
    start_time: float | None = None
    max_batches = warmup + batches
    print(
        "compile preflight: "
        f"warmup_batches={warmup} measure_batches={batches}"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (context, future) in enumerate(loader):
        if batch_index >= max_batches:
            break
        measuring = batch_index >= warmup
        if measuring and start_time is None:
            start_time = time.perf_counter()
        batch_start = time.perf_counter()
        try:
            context = move_to_device(context, device)
            future = move_to_device(future, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config):
                outputs = model(context, future)
                loss = temporal_graph_jepa_loss(outputs, config)["loss"]
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            scaler.scale(loss).backward()
        except torch.OutOfMemoryError as exc:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            msg = "compile preflight hit CUDA OOM; lower batch size or model size."
            raise RuntimeError(msg) from exc
        finally:
            optimizer.zero_grad(set_to_none=True)

        if measuring:
            measured_batches += 1
            measured_examples += int(getattr(context, "num_graphs", 1))
            _raise_if_batch_too_slow(time.perf_counter() - batch_start, train_cfg)

    if measured_batches == 0:
        return
    elapsed = time.perf_counter() - (start_time or time.perf_counter())
    batches_per_second = measured_batches / max(elapsed, 1e-9)
    examples_per_second = measured_examples / max(elapsed, 1e-9)
    estimated_epoch_seconds = len(loader) / max(batches_per_second, 1e-9)
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    row = {
        "event": "compile_preflight",
        "batches": measured_batches,
        "skipped_batches": skipped_batches,
        "batches_per_second": batches_per_second,
        "examples_per_second": examples_per_second,
        "estimated_epoch_seconds": estimated_epoch_seconds,
        "peak_memory_mb": peak_memory_mb,
    }
    if run_dir is not None:
        append_jsonl(run_dir / "perf.jsonl", row)
    print(
        "compile preflight: "
        f"{batches_per_second:.3f} batch/s, "
        f"epoch_eta={estimated_epoch_seconds / 60:.1f} min, "
        f"peak_vram={peak_memory_mb:.0f} MB"
    )
    min_bps = train_cfg.get("preflight_min_batches_per_second")
    if min_bps is not None and batches_per_second < float(min_bps):
        msg = (
            "compile preflight failed: "
            f"{batches_per_second:.3f} batch/s < required {float(min_bps):.3f}."
        )
        raise RuntimeError(msg)
    max_epoch_seconds = train_cfg.get("preflight_max_epoch_seconds")
    if max_epoch_seconds is not None and estimated_epoch_seconds > float(max_epoch_seconds):
        msg = (
            "compile preflight failed: "
            f"estimated epoch {estimated_epoch_seconds:.1f}s > "
            f"limit {float(max_epoch_seconds):.1f}s."
        )
        raise RuntimeError(msg)


def _raise_if_batch_too_slow(batch_seconds: float, train_cfg: dict) -> None:
    max_batch_seconds = train_cfg.get("max_batch_seconds")
    if max_batch_seconds is not None and batch_seconds > float(max_batch_seconds):
        msg = (
            "training health gate failed: "
            f"batch took {batch_seconds:.1f}s > limit {float(max_batch_seconds):.1f}s."
        )
        raise RuntimeError(msg)


def _raise_if_epoch_too_slow(epoch_seconds: float, train_cfg: dict) -> None:
    max_epoch_seconds = train_cfg.get("max_epoch_seconds")
    if max_epoch_seconds is not None and epoch_seconds > float(max_epoch_seconds):
        msg = (
            "training health gate failed: "
            f"epoch took {epoch_seconds:.1f}s > limit {float(max_epoch_seconds):.1f}s."
        )
        raise RuntimeError(msg)


def _evaluate(model: ParticleJEPA, loader, device: torch.device) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for context, future in loader:
            context = move_to_device(context, device)
            future = move_to_device(future, device)
            with autocast_context(device, {"train": {"precision": "fp16"}}):
                outputs = model(context, future)
                running += temporal_graph_jepa_loss(outputs, {"train": {}})["loss"].item()
    return running / max(len(loader), 1)
