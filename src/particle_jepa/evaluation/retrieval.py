from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def nearest_future_indices(query_latents: Tensor, future_latents: Tensor) -> Tensor:
    query = F.normalize(query_latents, dim=-1)
    future = F.normalize(future_latents, dim=-1)
    scores = query @ future.T
    return scores.argmax(dim=-1)


def retrieval_accuracy(query_latents: Tensor, future_latents: Tensor) -> Tensor:
    predicted = nearest_future_indices(query_latents, future_latents)
    target = torch.arange(query_latents.size(0), device=query_latents.device)
    return (predicted == target).float().mean()
