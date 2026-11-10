"""Per-client parameters of the data-generating process."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ClientParameters:
    """Parameters defining P_k(X, Y) for one client.

    X_k ~ N(mu_k, diag(scale_k^2));  P(Y=1 | x, k) = sigmoid(w_k^T x + b_k).
    ``b_k`` is calibrated so that the marginal positive rate equals ``positive_rate``.
    """

    client_id: str
    num_samples: int
    positive_rate: float
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("mean", "scale", "weights"):
            d[key] = np.asarray(d[key]).round(6).tolist()
        return d
