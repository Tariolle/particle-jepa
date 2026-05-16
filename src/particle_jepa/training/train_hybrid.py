from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from particle_jepa.models import HybridGNSJEPA
from particle_jepa.training.losses import HybridLoss


def train_hybrid(dataset, config: dict, device: torch.device) -> HybridGNSJEPA:
    model_cfg = config["model"]
    train_cfg = config["train"]
    model = HybridGNSJEPA(
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        message_passing_steps=model_cfg["message_passing_steps"],
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    criterion = HybridLoss(
        dynamics_weight=train_cfg.get("dynamics_loss_weight", 1.0),
        jepa_weight=train_cfg.get("jepa_loss_weight", 0.2),
    )

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running = 0.0
        for context, future in tqdm(loader, desc=f"hybrid epoch {epoch + 1}", leave=False):
            context = context.to(device)
            future = future.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(context, future)
            losses = criterion(outputs, context)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip_norm", 1.0))
            optimizer.step()
            running += losses["loss"].item()
        print(f"epoch={epoch + 1} loss={running / max(len(loader), 1):.6f}")
    return model
