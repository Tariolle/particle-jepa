from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn


def compile_model(model: nn.Module, config: dict[str, Any]) -> nn.Module:
    """Compile a model with the configured torch.compile settings."""
    train_cfg = config.get("train", {})
    if not train_cfg.get("compile", True):
        return model
    try:
        return torch.compile(
            model,
            mode=train_cfg.get("compile_mode", "reduce-overhead"),
            fullgraph=train_cfg.get("compile_fullgraph", False),
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


def strip_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}
