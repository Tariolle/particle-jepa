from __future__ import annotations

import gc
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
    strip_compiled_state_dict,
    unwrap_compiled_model,
)
from particle_jepa.utils.runs import append_jsonl


def train_jepa(
    dataset,
    config: dict,
    device: torch.device,
    val_dataset=None,
    run_dir=None,
    tracker=None,
    resume_state: dict | None = None,
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
    start_epoch = 0
    if resume_state is not None:
        model.load_state_dict(strip_compiled_state_dict(resume_state["model"]))
        start_epoch = int(resume_state.get("epoch", 0))
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
    if resume_state is not None and "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
    if resume_state is not None and "scaler" in resume_state:
        scaler.load_state_dict(resume_state["scaler"])
    if start_epoch > 0:
        print(f"resuming JEPA training from epoch {start_epoch}")
    _run_compile_preflight(model, loader, optimizer, scaler, config, device, run_dir)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        running_parts: dict[str, float] = {}
        skipped_batches = 0
        oom_batches = 0
        slow_batches = 0
        max_batch_seconds = 0.0
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
            except torch.OutOfMemoryError:
                skipped_batches += 1
                oom_batches += 1
                optimizer.zero_grad(set_to_none=True)
                _clear_cuda_cache(device)
                continue
            running += loss.item()
            for name, value in losses.items():
                if name == "loss":
                    continue
                running_parts[name] = running_parts.get(name, 0.0) + value.item()
            batch_seconds = time.perf_counter() - batch_start
            max_batch_seconds = max(max_batch_seconds, batch_seconds)
            if _is_batch_slow(batch_seconds, train_cfg):
                slow_batches += 1
        used_batches = max(seen_batches - skipped_batches, 1)
        train_loss = running / used_batches
        val_loss = _evaluate(model, val_loader, device, config) if val_loader is not None else None
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "skipped_batches": skipped_batches,
            "oom_batches": oom_batches,
            "slow_batches": slow_batches,
            "max_batch_seconds": max_batch_seconds,
            "epoch_seconds": epoch_seconds,
            "batches_per_second": seen_batches / max(epoch_seconds, 1e-9),
            **{name: value / used_batches for name, value in running_parts.items()},
        }
        if run_dir is not None:
            append_jsonl(run_dir / "logs.jsonl", row)
        if tracker is not None:
            tracker.log(row, step=epoch + 1)
        print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={val_loss}")
        if run_dir is not None and (epoch + 1) % train_cfg.get("checkpoint_every", 1) == 0:
            checkpoint = {
                "model": unwrap_compiled_model(model).state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            save_checkpoint(checkpoint, run_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt")
            save_checkpoint(checkpoint, run_dir / "checkpoints" / "last.pt")
        _raise_if_epoch_too_slow(epoch_seconds, train_cfg)
        _raise_if_too_many_ooms(oom_batches, seen_batches, train_cfg)
        if oom_batches > 0:
            _clear_cuda_cache(device)
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
            _raise_if_preflight_batch_too_slow(
                time.perf_counter() - batch_start, train_cfg
            )

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
    max_peak_memory_mb = train_cfg.get("preflight_max_peak_memory_mb")
    if max_peak_memory_mb is not None and peak_memory_mb > float(max_peak_memory_mb):
        msg = (
            "compile preflight failed: "
            f"peak VRAM {peak_memory_mb:.0f} MB > "
            f"limit {float(max_peak_memory_mb):.0f} MB."
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


def _is_batch_slow(batch_seconds: float, train_cfg: dict) -> bool:
    max_batch_seconds = train_cfg.get("slow_batch_seconds")
    return max_batch_seconds is not None and batch_seconds > float(max_batch_seconds)


def _raise_if_preflight_batch_too_slow(batch_seconds: float, train_cfg: dict) -> None:
    max_batch_seconds = train_cfg.get("preflight_max_batch_seconds")
    if max_batch_seconds is not None and batch_seconds > float(max_batch_seconds):
        msg = (
            "compile preflight failed: "
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


def _raise_if_too_many_ooms(oom_batches: int, seen_batches: int, train_cfg: dict) -> None:
    max_fraction = train_cfg.get("max_oom_fraction")
    if max_fraction is not None and seen_batches > 0:
        oom_fraction = oom_batches / seen_batches
        if oom_fraction > float(max_fraction):
            msg = (
                "training health gate failed: "
                f"OOM batch fraction {oom_fraction:.3f} > limit {float(max_fraction):.3f}."
            )
            raise RuntimeError(msg)


def _clear_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _evaluate(model: ParticleJEPA, loader, device: torch.device, config: dict) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for context, future in loader:
            context = move_to_device(context, device)
            future = move_to_device(future, device)
            with autocast_context(device, config):
                outputs = model(context, future)
                running += temporal_graph_jepa_loss(outputs, config)["loss"].item()
    return running / max(len(loader), 1)
