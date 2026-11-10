"""FedAvg (McMahan et al., 2017)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from federated_intelligence.server.aggregator import (
    apply_update,
    normalized_weights,
    weighted_average,
)

Params = list[np.ndarray]


class FedAvg:
    name = "fedavg"
    mu = 0.0

    def aggregate(
        self,
        global_params: Sequence[np.ndarray],
        deltas: Sequence[Sequence[np.ndarray]],
        num_samples: Sequence[float],
        server_lr: float = 1.0,
    ) -> Params:
        """w_{t+1} = w_t + lr * sum_k (n_k / N) delta_k  (== weight-space FedAvg for lr=1)."""
        avg = weighted_average(deltas, normalized_weights(num_samples))
        return apply_update(global_params, avg, server_lr)

    def describe(self) -> dict:
        return {"algorithm": self.name, "mu": self.mu}
