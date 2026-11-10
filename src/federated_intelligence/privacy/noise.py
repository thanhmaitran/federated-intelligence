"""Gaussian noise for the Gaussian mechanism."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def gaussian_noise_like(
    template: Sequence[np.ndarray], std: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Independent N(0, std^2) noise with the shapes of ``template`` (float32)."""
    return [rng.normal(0.0, std, size=t.shape).astype(np.float32) for t in template]
