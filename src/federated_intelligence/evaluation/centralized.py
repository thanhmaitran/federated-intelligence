"""Centralized reference baselines (Phase 6).

The centralized model sees the union of all client *training* splits - exactly the data the
federated run could use - and is trained with the same optimiser (``train_local``), model
architecture, preprocessing and evaluation protocol. It is an upper-bound reference, not a
privacy-preserving system.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from federated_intelligence.clients.local_trainer import LocalTrainingConfig, train_local
from federated_intelligence.data.federated_dataset import FederatedDataset
from federated_intelligence.evaluation.metrics import evaluate_model, perplexity
from federated_intelligence.models.factory import count_parameters, get_parameters, model_factory
from federated_intelligence.utils.seeds import derive_seed


def train_centralized(
    ds: FederatedDataset,
    model_name: str = "mlp",
    hidden: list[int] | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    momentum: float = 0.0,
    seed: int = 42,
    eval_every: int = 1,
) -> dict[str, Any]:
    """Train on pooled client-train data; returns per-epoch curves and final metrics."""
    kwargs: dict[str, Any] = {"num_features": ds.num_features, "num_classes": ds.num_classes, "init_seed": seed}
    if hidden is not None:
        kwargs["hidden"] = hidden
    if model_name == "gru":
        kwargs = {"vocab_size": ds.num_classes, "init_seed": seed}
    elif model_name == "cnn":
        kwargs = {"num_classes": ds.num_classes, "init_seed": seed}
    model = model_factory(model_name, **kwargs)()
    x, y = ds.pooled_train()
    vx, vy = ds.pooled_val()
    cfg = LocalTrainingConfig(1, batch_size, learning_rate, momentum, 0.0, 0.0)
    curve: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        res = train_local(model, x, y, cfg, get_parameters(model), derive_seed(seed, "central", ep))
        if ep % eval_every == 0 or ep == epochs:
            te = evaluate_model(model, ds.global_test_x, ds.global_test_y)
            va = evaluate_model(model, vx, vy)
            curve.append({"epoch": ep, "train_loss": res.train_loss,
                          **{f"test_{k}": v for k, v in te.items()}, **{f"val_{k}": v for k, v in va.items()}})
    train_seconds = time.perf_counter() - t0
    per_client = [evaluate_model(model, c.x_test, c.y_test)["accuracy"] for c in ds.clients.values()]
    final = curve[-1]
    out: dict[str, Any] = {
        "model": model_name,
        "hidden": hidden,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "num_parameters": count_parameters(model),
        "train_samples": int(len(y)),
        "train_seconds": train_seconds,
        "final": {k: v for k, v in final.items() if k != "epoch"},
        "per_client_test_accuracy": {"mean": float(np.mean(per_client)), "std": float(np.std(per_client)),
                                     "min": float(np.min(per_client))},
        "curve": curve,
    }
    if model_name == "gru":
        out["final"]["test_perplexity"] = perplexity(out["final"]["test_loss"])
    return out
