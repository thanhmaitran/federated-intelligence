"""IID partitioning: a uniformly random, near-equal-size split."""

from __future__ import annotations

import numpy as np

from federated_intelligence.data.partitioners.base import BasePartitioner, Partition


class IIDPartitioner(BasePartitioner):
    method = "iid"

    def partition(self, labels: np.ndarray) -> Partition:
        perm = self.rng.permutation(len(labels))
        return self._to_partition(np.array_split(perm, self.num_clients))
