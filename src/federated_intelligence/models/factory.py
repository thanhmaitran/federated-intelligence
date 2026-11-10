"""Model factory and parameter <-> NumPy conversion helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn

from federated_intelligence.models.nlp.gru import CharGRU
from federated_intelligence.models.tabular.models import MLP, LogisticRegression
from federated_intelligence.models.vision.cnn import FemnistCNN


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("logreg", "logistic_regression"):
        return LogisticRegression(kwargs["num_features"], kwargs.get("num_classes", 2))
    if name == "mlp":
        hidden = tuple(kwargs.get("hidden", (64, 32)))
        return MLP(kwargs["num_features"], kwargs.get("num_classes", 2), hidden)
    if name == "cnn":
        return FemnistCNN(kwargs.get("num_classes", 62))
    if name == "gru":
        return CharGRU(kwargs.get("vocab_size", 80))
    raise ValueError(f"Unknown model '{name}'")


def model_factory(name: str, **kwargs) -> Callable[[], nn.Module]:
    """Factory with *identical initial weights* on every call (seeded by ``init_seed``)."""
    init_seed = kwargs.pop("init_seed", 0)

    def make() -> nn.Module:
        with torch.random.fork_rng():
            torch.manual_seed(init_seed)
            return build_model(name, **kwargs)

    return make


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """All trainable and buffer state as float32 NumPy arrays (state_dict order)."""
    return [v.detach().cpu().numpy().astype(np.float32, copy=True) for v in model.state_dict().values()]


def set_parameters(model: nn.Module, params: Sequence[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(params):
        raise ValueError(f"expected {len(keys)} tensors, got {len(params)}")
    state = {k: torch.as_tensor(np.asarray(p)) for k, p in zip(keys, params, strict=True)}
    model.load_state_dict(state, strict=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
