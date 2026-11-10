"""Aggregation engine with update validation.

FedAvg in parameter space,  w_{t+1} = sum_k (n_k / N) w^{(k)},  is identical to applying the
sample-weighted average of the *deltas* to the global model, w_{t+1} = w_t + sum_k (n_k / N) d_k.
The coordinator works with deltas (needed for compression and DP); ``fedavg_params`` is the
direct weight-space form used for invariant tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Params = list[np.ndarray]


class AggregationError(ValueError):
    pass


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""


def validate_update(update: Sequence[np.ndarray], template: Sequence[np.ndarray]) -> ValidationResult:
    """Shape, dtype and finiteness checks for one client update."""
    if len(update) != len(template):
        return ValidationResult(False, f"expected {len(template)} tensors, got {len(update)}")
    for i, (u, t) in enumerate(zip(update, template, strict=True)):
        u = np.asarray(u)
        if u.shape != t.shape:
            return ValidationResult(False, f"tensor {i}: shape {u.shape} != {t.shape}")
        if not np.issubdtype(u.dtype, np.floating):
            return ValidationResult(False, f"tensor {i}: non-floating dtype {u.dtype}")
        if not np.all(np.isfinite(u)):
            return ValidationResult(False, f"tensor {i}: non-finite values")
    return ValidationResult(True)


def normalized_weights(num_samples: Sequence[float]) -> np.ndarray:
    n = np.asarray(num_samples, dtype=np.float64)
    if n.size == 0:
        raise AggregationError("cannot aggregate an empty set of updates")
    if np.any(n < 0) or not np.all(np.isfinite(n)):
        raise AggregationError("invalid sample counts")
    total = n.sum()
    if total <= 0:
        raise AggregationError("total sample count must be positive")
    return n / total


def weighted_average(updates: Sequence[Sequence[np.ndarray]], weights: np.ndarray) -> Params:
    """sum_k weights[k] * updates[k], accumulated in float64 and returned as float32."""
    if len(updates) == 0:
        raise AggregationError("cannot aggregate an empty set of updates")
    if len(updates) != len(weights):
        raise AggregationError("updates and weights differ in length")
    if not np.isclose(np.sum(weights), 1.0):
        raise AggregationError("aggregation weights must sum to 1")
    out = [np.zeros(t.shape, dtype=np.float64) for t in updates[0]]
    for w, upd in zip(weights, updates, strict=True):
        for acc, u in zip(out, upd, strict=True):
            acc += w * np.asarray(u, dtype=np.float64)
    return [o.astype(np.float32) for o in out]


def fedavg_params(client_params: Sequence[Sequence[np.ndarray]], num_samples: Sequence[float]) -> Params:
    """Direct FedAvg of client weights."""
    return weighted_average(client_params, normalized_weights(num_samples))


def apply_update(
    global_params: Sequence[np.ndarray], avg_delta: Sequence[np.ndarray], server_lr: float = 1.0
) -> Params:
    return [
        (np.asarray(g, dtype=np.float32) + np.float32(server_lr) * np.asarray(d, dtype=np.float32))
        for g, d in zip(global_params, avg_delta, strict=True)
    ]
