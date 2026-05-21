from __future__ import annotations

import math
import random
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.utils.data import Subset
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
        if train_cfg.get("max_autotune_gemm_backends") is not None:
            inductor_config.max_autotune_gemm_backends = train_cfg[
                "max_autotune_gemm_backends"
            ]
    except Exception as exc:
        print(f"could not apply torch._inductor dynamic graph settings: {exc}")


def compile_model(model: nn.Module, config: dict[str, Any]) -> nn.Module:
    """Compile a model with the configured torch.compile settings."""
    train_cfg = config.get("train", {})
    if not train_cfg.get("compile", True):
        return model
    configure_inductor(config)
    scope = train_cfg.get("compile_scope", "full")
    kwargs = _compile_kwargs(train_cfg)
    if scope == "regional":
        compile_regions = getattr(model, "compile_regions", None)
        if compile_regions is None:
            msg = f"{model.__class__.__name__} does not support regional compilation."
            raise RuntimeError(msg)
        _set_regional_cudagraphs(train_cfg)
        kwargs = _regional_compile_kwargs(train_cfg)
        input_dtype = _regional_input_dtype(train_cfg)
        count = compile_regions(input_dtype=input_dtype, **kwargs)
        print(f"torch.compile regional scope: compiled {count} dense regions")
        return model
    if scope != "full":
        msg = f"Unsupported compile_scope '{scope}'. Expected 'full' or 'regional'."
        raise ValueError(msg)
    try:
        return torch.compile(model, **kwargs)
    except Exception as exc:
        if train_cfg.get("compile_required", True):
            msg = "torch.compile failed and compile_required=true."
            raise RuntimeError(msg) from exc
        print(f"torch.compile unavailable, continuing eager: {exc}")
        return model


def _compile_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "mode": train_cfg.get("compile_mode", "reduce-overhead"),
        "fullgraph": train_cfg.get("compile_fullgraph", False),
    }
    if train_cfg.get("compile_dynamic") is not None:
        kwargs["dynamic"] = bool(train_cfg["compile_dynamic"])
    return kwargs


def _regional_compile_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = _compile_kwargs(train_cfg)
    if train_cfg.get("regional_compile_mode") is not None:
        kwargs["mode"] = train_cfg["regional_compile_mode"]
    regional_dynamic = train_cfg.get("regional_compile_dynamic", True)
    if regional_dynamic is None:
        kwargs.pop("dynamic", None)
    else:
        kwargs["dynamic"] = bool(regional_dynamic)
    return kwargs


def _set_regional_cudagraphs(train_cfg: dict[str, Any]) -> None:
    if "regional_compile_cudagraphs" not in train_cfg:
        return
    try:
        torch._inductor.config.triton.cudagraphs = bool(
            train_cfg["regional_compile_cudagraphs"]
        )
    except Exception as exc:
        print(f"could not set regional torch.compile cudagraphs: {exc}")


def _regional_input_dtype(train_cfg: dict[str, Any]) -> torch.dtype | None:
    precision = train_cfg.get("precision", "fp16")
    if precision == "fp16" and torch.cuda.is_available():
        return torch.float16
    return None


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
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    if bool(data_cfg.get("bucketed_batches", False)):
        kwargs["batch_sampler"] = BucketBatchSampler(
            dataset=dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=shuffle,
            bucket_size=int(data_cfg.get("bucket_size", 256)),
            drop_last=bool(data_cfg.get("drop_last", False)),
            seed=int(config.get("seed", 7)),
        )
    else:
        kwargs["batch_size"] = int(train_cfg["batch_size"])
        kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **kwargs)


def move_to_device(batch, device: torch.device):
    """Move a PyG batch to device using non-blocking copies when pinned memory is enabled."""
    return batch.to(device, non_blocking=device.type == "cuda")


def strip_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    regional_segment = ".module._orig_mod."
    if not any(key.startswith(prefix) or regional_segment in key for key in state_dict):
        return state_dict
    return {
        key.removeprefix(prefix).replace(regional_segment, "."): value
        for key, value in state_dict.items()
    }


class BucketBatchSampler:
    """Batch nearby graph sizes together to reduce dynamic-shape churn."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool,
        bucket_size: int = 256,
        drop_last: bool = False,
        seed: int = 7,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(bucket_size, batch_size)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._sizes = [_estimate_sample_size(dataset, index) for index in range(len(dataset))]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)
        windows = [
            sorted(indices[start : start + self.bucket_size], key=self._sizes.__getitem__)
            for start in range(0, len(indices), self.bucket_size)
        ]
        if self.shuffle:
            rng.shuffle(windows)
        for window in windows:
            for start in range(0, len(window), self.batch_size):
                batch = window[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


def _estimate_sample_size(dataset, index: int) -> int:
    if isinstance(dataset, Subset):
        return _estimate_sample_size(dataset.dataset, int(dataset.indices[index]))
    estimator = getattr(dataset, "estimate_graph_size", None)
    if estimator is not None:
        return int(estimator(index))
    return 0
