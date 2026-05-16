from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def nearest_future_indices(query_latents: Tensor, future_latents: Tensor, top_k: int = 1) -> Tensor:
    query = F.normalize(query_latents, dim=-1)
    future = F.normalize(future_latents, dim=-1)
    scores = query @ future.T
    return scores.topk(k=top_k, dim=-1).indices


def retrieval_accuracy(query_latents: Tensor, future_latents: Tensor, top_k: int = 1) -> Tensor:
    predicted = nearest_future_indices(query_latents, future_latents, top_k=top_k)
    target = torch.arange(query_latents.size(0), device=query_latents.device)
    return (predicted == target[:, None]).any(dim=-1).float().mean()


def chance_retrieval_accuracy(num_samples: int, top_k: int = 1) -> float:
    if num_samples <= 0:
        return 0.0
    return min(top_k, num_samples) / num_samples


def random_latent_retrieval_accuracy(
    future_latents: Tensor,
    top_k: int = 1,
    trials: int = 32,
    seed: int = 0,
) -> Tensor:
    generator = torch.Generator(device=future_latents.device).manual_seed(seed)
    scores = []
    for _ in range(trials):
        random_queries = torch.randn(
            future_latents.shape,
            generator=generator,
            device=future_latents.device,
            dtype=future_latents.dtype,
        )
        scores.append(retrieval_accuracy(random_queries, future_latents, top_k=top_k))
    return torch.stack(scores).mean()
