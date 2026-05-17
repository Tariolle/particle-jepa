from __future__ import annotations

import torch
from tqdm import tqdm

from particle_jepa.models import ParticleJEPA
from particle_jepa.training.losses import temporal_graph_jepa_loss
from particle_jepa.utils.perf import (
    autocast_context,
    compile_model,
    make_grad_scaler,
    make_pyg_dataloader,
    move_to_device,
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

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running = 0.0
        running_parts: dict[str, float] = {}
        skipped_batches = 0
        seen_batches = 0
        for context, future in tqdm(loader, desc=f"jepa epoch {epoch + 1}", leave=False):
            seen_batches += 1
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
            running += loss.item()
            for name, value in losses.items():
                if name == "loss":
                    continue
                running_parts[name] = running_parts.get(name, 0.0) + value.item()
        used_batches = max(seen_batches - skipped_batches, 1)
        train_loss = running / used_batches
        val_loss = _evaluate(model, val_loader, device) if val_loader is not None else None
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "skipped_batches": skipped_batches,
            **{name: value / used_batches for name, value in running_parts.items()},
        }
        if run_dir is not None:
            append_jsonl(run_dir / "logs.jsonl", row)
        if tracker is not None:
            tracker.log(row, step=epoch + 1)
        print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={val_loss}")
    return model


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
