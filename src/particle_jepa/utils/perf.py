from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch_geometric.loader import DataLoader


def configure_inductor(config: dict[str, Any]) -> None:
    """Apply torch.compile settings that matter for variable-size PyG graphs."""
    train_cfg = config.get("train", {})
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        try:
            torch.set_float32_matmul_precision(train_cfg.get("float32_matmul_precision", "high"))
        except Exception as exc:
            print(f"could not set float32 matmul precision: {exc}")
    if not train_cfg.get("compile", True):
        return
    try:
        inductor_config = torch._inductor.config
        triton_config = inductor_config.triton
        triton_config.cudagraph_skip_dynamic_graphs = train_cfg.get(
            "cudagraph_skip_dynamic_graphs", True
        )
        warn_limit = train_cfg.get("cudagraph_dynamic_shape_warn_limit", None)
        triton_config.cudagraph_dynamic_shape_warn_limit = warn_limit
    except Exception as exc:
        print(f"could not apply torch._inductor dynamic graph settings: {exc}")


def compile_model(model: nn.Module, config: dict[str, Any]) -> nn.Module:
    """Compile a model with the configured torch.compile settings."""
    train_cfg = config.get("train", {})
    if not train_cfg.get("compile", True):
        return model
    configure_inductor(config)
    try:
        kwargs = {}
        if train_cfg.get("compile_dynamic", None) is not None:
            kwargs["dynamic"] = bool(train_cfg["compile_dynamic"])
        return torch.compile(
            model,
            mode=train_cfg.get("compile_mode", "reduce-overhead"),
            fullgraph=train_cfg.get("compile_fullgraph", False),
            **kwargs,
        )
    except Exception as exc:
        print(f"torch.compile unavailable, continuing eager: {exc}")
        return model


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def autocast_context(device: torch.device, config: dict[str, Any]):
    """Return an FP16 autocast context when CUDA is available."""
    precision = config.get("train", {}).get("precision", "fp16")
    if precision != "fp16":
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def make_grad_scaler(device: torch.device, config: dict[str, Any]) -> torch.amp.GradScaler:
    enabled = config.get("train", {}).get("precision", "fp16") == "fp16" and device.type == "cuda"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def make_pyg_dataloader(
    dataset,
    config: dict[str, Any],
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    """Create a PyG DataLoader with CUDA-friendly host-side settings."""
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    num_workers = int(data_cfg.get("num_workers", train_cfg.get("num_workers", 0)))
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def move_to_device(batch, device: torch.device):
    """Move a PyG batch to device using non-blocking copies when pinned memory is enabled."""
    return batch.to(device, non_blocking=device.type == "cuda")


def strip_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}
