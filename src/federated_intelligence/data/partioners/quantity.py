"""Quantity-skew partitioning with log-normal client sizes (labels stay IID)."""

from __future__ import annotations

import numpy as np

from federated_intelligence.data.partitioners.base import BasePartitioner, Partition


class QuantitySkewPartitioner(BasePartitioner):
    method = "quantity_skew"

    def __init__(
        self, num_clients: int, sigma: float = 1.0, seed: int = 42, min_samples: int = 10
    ):
        super().__init__(num_clients, seed)
        self.sigma = sigma
        self.min_samples = min_samples

    def params(self) -> dict:
        return {"sigma": self.sigma, "min_samples": self.min_samples}

    def partition(self, labels: np.ndarray) -> Partition:
        n = len(labels)
        if n < self.num_clients * self.min_samples:
            raise ValueError("Not enough samples for the requested minimum client size")
        rng = self.rng
        weights = rng.lognormal(0.0, self.sigma, size=self.num_clients)
        weights /= weights.sum()
        spare = n - self.num_clients * self.min_samples
        sizes = self.min_samples + np.floor(weights * spare).astype(int)
        # Hand the rounding remainder to the largest clients so the total is exact.
        remainder = n - sizes.sum()
        for k in np.argsort(-weights)[:remainder]:
            sizes[k] += 1
        perm = rng.permutation(n)
        cuts = np.cumsum(sizes)[:-1]
        return self._to_partition(np.split(perm, cuts))
