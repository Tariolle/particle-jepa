from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from particle_jepa.models import ParticleJEPA
from particle_jepa.training.losses import jepa_loss
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
    ).to(device)
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

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running = 0.0
        for context, future in tqdm(loader, desc=f"jepa epoch {epoch + 1}", leave=False):
            context = context.to(device)
            future = future.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(context, future)
            loss = jepa_loss(outputs["prediction"], outputs["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip_norm", 1.0))
            optimizer.step()
            running += loss.item()
        train_loss = running / max(len(loader), 1)
        val_loss = _evaluate(model, val_loader, device) if val_loader is not None else None
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
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
            context = context.to(device)
            future = future.to(device)
            outputs = model(context, future)
            running += jepa_loss(outputs["prediction"], outputs["target"]).item()
    return running / max(len(loader), 1)
