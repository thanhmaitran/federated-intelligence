"""Deterministic seeding helpers.

Every stochastic component receives an explicit ``numpy.random.Generator`` or an
integer seed derived from a single experiment seed, so that results do not depend
on execution order or on the executor backend (local vs. Ray).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch global RNGs (used only as a safety net)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - safety net for third-party code
    torch.manual_seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def derive_seed(*parts: int | str) -> int:
    """Derive a stable 32-bit seed from heterogeneous parts.

    Used to give every (experiment seed, round, client) triple an independent but
    reproducible stream, regardless of the order in which clients are executed.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
