from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from particle_jepa.models import GraphNetworkSimulator
from particle_jepa.training.losses import acceleration_loss
from particle_jepa.utils.perf import (
    autocast_context,
    compile_model,
    make_grad_scaler,
    make_pyg_dataloader,
    move_to_device,
)
from particle_jepa.utils.runs import append_jsonl


def train_gns(
    dataset, config: dict, device: torch.device, val_dataset=None, run_dir=None, tracker=None
) -> GraphNetworkSimulator:
    model_cfg = config["model"]
    train_cfg = config["train"]
    model = GraphNetworkSimulator(
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        message_passing_steps=model_cfg["message_passing_steps"],
        dropout=model_cfg.get("dropout", 0.0),
        mlp_layers=model_cfg.get("mlp_layers", 2),
    ).to(device)
    model = compile_model(model, config)
    loader = make_pyg_dataloader(_ContextOnlyDataset(dataset), config, device, shuffle=True)
    val_loader = (
        make_pyg_dataloader(_ContextOnlyDataset(val_dataset), config, device, shuffle=False)
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
        for context in tqdm(loader, desc=f"gns epoch {epoch + 1}", leave=False):
            context = move_to_device(context, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config):
                prediction = model(context)
                loss = acceleration_loss(
                    prediction, context.y_acceleration, getattr(context, "dynamic_mask", None)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_loss = running / max(len(loader), 1)
        val_loss = _evaluate(model, val_loader, device, config) if val_loader is not None else None
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        if run_dir is not None:
            append_jsonl(run_dir / "logs.jsonl", row)
        if tracker is not None:
            tracker.log(row, step=epoch + 1)
        print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={val_loss}")
    return model


def _evaluate(
    model: GraphNetworkSimulator, loader, device: torch.device, config: dict
) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for context in loader:
            context = move_to_device(context, device)
            with autocast_context(device, config):
                running += acceleration_loss(
                    model(context), context.y_acceleration, getattr(context, "dynamic_mask", None)
                ).item()
    return running / max(len(loader), 1)


class _ContextOnlyDataset(Dataset):
    """Adapter that lets GNS skip constructing unused future graphs."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return _context_at(self.dataset, int(index))

    def estimate_graph_size(self, index: int) -> int:
        return _estimate_graph_size(self.dataset, int(index))


def _context_at(dataset, index: int):
    if isinstance(dataset, Subset):
        return _context_at(dataset.dataset, int(dataset.indices[index]))
    context_at = getattr(dataset, "context_at", None)
    if context_at is not None:
        return context_at(index)
    return dataset[index][0]


def _estimate_graph_size(dataset, index: int) -> int:
    if isinstance(dataset, Subset):
        return _estimate_graph_size(dataset.dataset, int(dataset.indices[index]))
    estimator = getattr(dataset, "estimate_graph_size", None)
    if estimator is None:
        return 0
    return int(estimator(index))
