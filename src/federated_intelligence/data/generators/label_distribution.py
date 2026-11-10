"""Controls the class proportions of each client."""

from __future__ import annotations

import numpy as np

from federated_intelligence.data.generators.generator_config import GeneratorConfig


def sample_positive_rates(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Return p_k for every client.

    Skewed scenarios draw p_k ~ Beta(a, a); with a < 1 the mass moves towards 0/1 so
    clients see very different class priors. Otherwise all clients share one rate.
    """
    if cfg.scenario.skews_labels:
        p = rng.beta(cfg.label_beta, cfg.label_beta, size=cfg.num_clients)
        return np.clip(p, 0.02, 0.98)
    return np.full(cfg.num_clients, cfg.balanced_positive_rate)


def sample_sizes(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Number of samples per client (quantity skew uses a log-normal law)."""
    lo, hi = cfg.min_samples_per_client, cfg.max_samples_per_client
    if cfg.scenario.skews_quantity:
        raw = rng.lognormal(mean=0.0, sigma=cfg.quantity_sigma, size=cfg.num_clients)
        # Min-max map to [lo, hi] preserves the heavy-tailed shape of the draw.
        scaled = lo + (raw - raw.min()) / max(raw.max() - raw.min(), 1e-12) * (hi - lo)
        return np.round(scaled).astype(int)
    return np.full(cfg.num_clients, (lo + hi) // 2, dtype=int)


def calibrate_bias(
    x: np.ndarray, weights: np.ndarray, target_rate: float, tol: float = 1e-4
) -> float:
    """Find b so that mean(sigmoid(x @ w + b)) == target_rate (bisection)."""
    z = x @ weights
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(z + mid)))))
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
