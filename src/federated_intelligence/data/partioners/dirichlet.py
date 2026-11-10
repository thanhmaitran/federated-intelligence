"""Dirichlet label-skew partitioning.

For every class c, draw proportions ``p_c ~ Dirichlet(alpha * 1_K)`` over the K clients and
distribute the samples of class c accordingly. Small alpha concentrates each class on a few
clients (strong heterogeneity); large alpha approaches IID.

A minimum-size constraint is enforced by re-drawing (bounded retries) so that no client is
empty, which would make local training undefined. For very small alpha and many clients the
constraint is almost never met by rejection sampling alone; in that case a deterministic
*repair* step moves the minimum number of samples from the largest clients to the undersized
ones. The number of repaired clients is recorded (``repaired_clients``) so that the departure
from a pure Dirichlet draw is visible in the manifest.
"""

from __future__ import annotations

import numpy as np

from federated_intelligence.data.partitioners.base import BasePartitioner, Partition


class DirichletPartitioner(BasePartitioner):
    method = "dirichlet"

    def __init__(
        self,
        num_clients: int,
        alpha: float,
        seed: int = 42,
        min_samples_per_client: int = 10,
        max_attempts: int = 200,
    ):
        super().__init__(num_clients, seed)
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.min_samples = min_samples_per_client
        self.max_attempts = max_attempts
        self.repaired_clients = 0

    def params(self) -> dict:
        return {
            "alpha": self.alpha,
            "min_samples_per_client": self.min_samples,
            "repaired_clients": self.repaired_clients,
        }

    def partition(self, labels: np.ndarray) -> Partition:
        labels = np.asarray(labels)
        classes = np.unique(labels)
        rng = self.rng
        per_class = []
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            per_class.append(idx)

        best: list[list[int]] | None = None
        best_min = -1
        for _ in range(self.max_attempts):
            # Draw proportions and derive integer cut points first; only materialize the
            # assignment once the minimum-size constraint is met (cheap rejection sampling).
            cut_sets: list[np.ndarray] = []
            sizes = np.zeros(self.num_clients, dtype=int)
            for idx in per_class:
                props = rng.dirichlet(np.full(self.num_clients, self.alpha))
                cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
                cut_sets.append(cuts)
                sizes += np.diff(np.concatenate([[0], cuts, [len(idx)]]))
            if sizes.min() > best_min:
                best_min = int(sizes.min())
                buckets: list[list[int]] = [[] for _ in range(self.num_clients)]
                for idx, cuts in zip(per_class, cut_sets, strict=True):
                    for k, part in enumerate(np.split(idx, cuts)):
                        buckets[k].extend(part.tolist())
                best = buckets
            if best_min >= self.min_samples:
                break
        assert best is not None
        if best_min < self.min_samples:
            self._repair(best)
        return self._to_partition(best)

    def _repair(self, buckets: list[list[int]]) -> None:
        """Top up undersized clients from the currently largest client (deterministic)."""
        total = sum(len(b) for b in buckets)
        if total < self.min_samples * self.num_clients:
            raise RuntimeError("Not enough samples to give every client min_samples_per_client")
        repaired = 0
        for k in np.argsort([len(b) for b in buckets]):
            need = self.min_samples - len(buckets[k])
            if need <= 0:
                continue
            donor = int(np.argmax([len(b) for b in buckets]))
            moved = [buckets[donor].pop() for _ in range(need)]
            buckets[k].extend(moved)
            repaired += 1
        self.repaired_clients = repaired
