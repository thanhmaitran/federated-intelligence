"""Low-level quantization primitives."""

from __future__ import annotations

import numpy as np


def to_float16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)


def from_float16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """Symmetric per-tensor 8-bit quantization: q = round(x / s), s = max|x| / 127."""
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    scale = np.float32(max_abs / 127.0 if max_abs > 0 else 1.0)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: np.float32) -> np.ndarray:
    return q.astype(np.float32) * np.float32(scale)
