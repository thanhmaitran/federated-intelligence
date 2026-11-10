"""Convergence analysis of round-wise metric curves."""

from __future__ import annotations

import numpy as np


def rounds_to_target(rounds: list[int], values: list[float], target: float) -> int | None:
    """First round whose metric reaches ``target`` (None if never)."""
    for r, v in zip(rounds, values, strict=True):
        if v >= target:
            return r
    return None


def area_under_curve(rounds: list[int], values: list[float]) -> float:
    """Normalised AUC of the metric curve (higher = faster convergence)."""
    r = np.asarray(rounds, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(r) < 2:
        return float(v[0]) if len(v) else float("nan")
    return float(np.trapezoid(v, r) / (r[-1] - r[0]))


def curve_volatility(values: list[float], tail: int = 10) -> float:
    """Std of the last ``tail`` values: round-to-round instability (client drift symptom)."""
    return float(np.std(values[-tail:])) if values else float("nan")


def final_plateau(values: list[float], tail: int = 5) -> float:
    return float(np.mean(values[-tail:])) if values else float("nan")
