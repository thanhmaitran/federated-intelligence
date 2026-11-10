"""Local training on one client (SGD with optional FedProx proximal term)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class LocalTrainingConfig:
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.05
    momentum: float = 0.0
    weight_decay: float = 0.0
    mu: float = 0.0  # FedProx proximal coefficient; 0 => plain local SGD (FedAvg)


@dataclass
class LocalTrainingResult:
    train_loss: float
    num_steps: int
    num_samples: int
    seconds: float


def proximal_term(model: nn.Module, anchor: dict[str, torch.Tensor]) -> torch.Tensor:
    """(1/2) * ||w - w_t||_2^2 over trainable parameters (``mu`` is applied by the caller)."""
    total = torch.zeros((), dtype=torch.float32)
    for name, p in model.named_parameters():
        total = total + torch.sum((p - anchor[name]) ** 2)
    return 0.5 * total


def train_local(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    cfg: LocalTrainingConfig,
    global_params: Sequence[np.ndarray],
    seed: int,
) -> LocalTrainingResult:
    """Run ``cfg.local_epochs`` epochs of mini-batch SGD starting from ``global_params``.

    The objective is  L(w) + (mu/2) ||w - w_t||^2  where w_t = ``global_params`` (fixed anchor).
    Shuffling uses a dedicated ``torch.Generator`` so results do not depend on global RNG state
    or on which executor runs the client.
    """
    start = time.perf_counter()
    keys = list(model.state_dict().keys())
    anchor_all = {k: torch.as_tensor(np.asarray(p)) for k, p in zip(keys, global_params, strict=True)}
    anchor = {name: anchor_all[name] for name, _ in model.named_parameters()}

    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    n = len(yt)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    steps, running, seen = 0, 0.0, 0
    for _ in range(cfg.local_epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            # proximal_term already carries the 1/2, so mu * term == (mu/2) * ||w - w_t||^2.
            total = loss + cfg.mu * proximal_term(model, anchor) if cfg.mu > 0 else loss
            total.backward()
            opt.step()
            running += float(loss.detach()) * len(idx)
            seen += len(idx)
            steps += 1
    return LocalTrainingResult(
        train_loss=running / max(seen, 1),
        num_steps=steps,
        num_samples=n,
        seconds=time.perf_counter() - start,
    )
