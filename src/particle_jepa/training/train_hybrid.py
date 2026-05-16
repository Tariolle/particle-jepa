from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from particle_jepa.models import HybridGNSJEPA
from particle_jepa.training.losses import HybridLoss
from particle_jepa.utils.perf import autocast_context, compile_model, make_grad_scaler
from particle_jepa.utils.runs import append_jsonl


def train_hybrid(
    dataset, config: dict, device: torch.device, val_dataset=None, run_dir=None, tracker=None
) -> HybridGNSJEPA:
    model_cfg = config["model"]
    train_cfg = config["train"]
    model = HybridGNSJEPA(
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        message_passing_steps=model_cfg["message_passing_steps"],
        dropout=model_cfg.get("dropout", 0.0),
        mlp_layers=model_cfg.get("mlp_layers", 2),
        max_horizon=model_cfg.get("max_horizon", 32),
        latent_predictor_steps=model_cfg.get("latent_predictor_steps", 2),
    ).to(device)
    model = compile_model(model, config)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)
        if val_dataset is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    criterion = HybridLoss(
        dynamics_weight=train_cfg.get("dynamics_loss_weight", 1.0),
        jepa_weight=train_cfg.get("jepa_loss_weight", 0.2),
        node_weight=train_cfg.get("node_prediction_weight", 1.0),
        sigreg_weight=train_cfg.get("sigreg_weight", 0.05),
        sigreg_sketch_dim=train_cfg.get("sigreg_sketch_dim", 64),
    )
    scaler = make_grad_scaler(device, config)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running = 0.0
        for context, future in tqdm(loader, desc=f"hybrid epoch {epoch + 1}", leave=False):
            context = context.to(device)
            future = future.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config):
                outputs = model(context, future)
                losses = criterion(outputs, context)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
            running += losses["loss"].item()
        train_loss = running / max(len(loader), 1)
        val_loss = (
            _evaluate(model, val_loader, criterion, device) if val_loader is not None else None
        )
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        if run_dir is not None:
            append_jsonl(run_dir / "logs.jsonl", row)
        if tracker is not None:
            tracker.log(row, step=epoch + 1)
        print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={val_loss}")
    return model


def _evaluate(model: HybridGNSJEPA, loader, criterion: HybridLoss, device: torch.device) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for context, future in loader:
            context = context.to(device)
            future = future.to(device)
            with autocast_context(device, {"train": {"precision": "fp16"}}):
                running += criterion(model(context, future), context)["loss"].item()
    return running / max(len(loader), 1)
