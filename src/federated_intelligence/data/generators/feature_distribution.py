"""Controls the feature distribution of each client."""

from __future__ import annotations

import numpy as np

from federated_intelligence.data.generators.generator_config import GeneratorConfig


def sample_feature_params(
    cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (means, scales) of shape (K, d).

    mu_k = mu_0 + Delta_k with mu_0 = 0 and Delta_k ~ N(0, shift^2 I) when feature skew is
    active; Sigma_k = diag(scale_k^2) with scale_k ~ U(range). Otherwise N(0, I) for all.
    """
    k, d = cfg.num_clients, cfg.num_features
    if cfg.scenario.skews_features:
        means = rng.normal(0.0, cfg.feature_shift, size=(k, d))
        lo, hi = cfg.feature_scale_range
        scales = rng.uniform(lo, hi, size=(k, d))
    else:
        means = np.zeros((k, d))
        scales = np.ones((k, d))
    return means, scales


def sample_features(
    mean: np.ndarray, scale: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.normal(loc=mean, scale=scale, size=(n, mean.shape[0]))
