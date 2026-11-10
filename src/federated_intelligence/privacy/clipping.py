"""L2 clipping of model updates (the sensitivity-bounding step of client-level DP)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Params = list[np.ndarray]


def global_l2_norm(update: Sequence[np.ndarray]) -> float:
    """L2 norm of the update viewed as one flat vector (float64 accumulation)."""
    return float(np.sqrt(sum(float(np.sum(np.square(u, dtype=np.float64))) for u in update)))


def clip_update(update: Sequence[np.ndarray], clip_norm: float) -> tuple[Params, float, bool]:
    """Return (clipped update, original norm, was_clipped).

    clipped = update * min(1, C / ||update||_2)
    """
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    norm = global_l2_norm(update)
    factor = min(1.0, clip_norm / norm) if norm > 0 else 1.0
    clipped = [(u * np.float32(factor)).astype(np.float32) for u in update]
    return clipped, norm, factor < 1.0
