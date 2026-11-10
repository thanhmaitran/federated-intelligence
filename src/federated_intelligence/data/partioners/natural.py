"""Natural partitioning by an existing group identifier.

FEMNIST: ``writer_id``. Shakespeare: ``character_id``. Group-wise splitting is also used to
create *writer-aware* train/test splits for unseen-writer generalization.
"""

from __future__ import annotations

import numpy as np

from federated_intelligence.data.partitioners.base import BasePartitioner, Partition


class NaturalPartitioner(BasePartitioner):
    method = "natural"

    def __init__(self, groups: np.ndarray, seed: int = 42, min_samples: int = 1):
        self.groups = np.asarray(groups)
        self.min_samples = min_samples
        super().__init__(len(np.unique(self.groups)), seed)

    def partition(self, labels: np.ndarray) -> Partition:
        if len(labels) != len(self.groups):
            raise ValueError("labels and groups must have equal length")
        out: Partition = {}
        for g in np.unique(self.groups):
            idx = np.flatnonzero(self.groups == g)
            if len(idx) >= self.min_samples:
                out[f"client_{g}"] = idx
        return out


def split_groups(
    groups: np.ndarray, holdout_fraction: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split *group ids* (e.g. writers) into seen/unseen sets - no group appears in both."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_hold = max(1, int(round(len(uniq) * holdout_fraction)))
    return np.sort(uniq[n_hold:]), np.sort(uniq[:n_hold])


def contiguous_split_with_gap(
    n: int, fractions: tuple[float, float, float], gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``n`` *ordered* items into contiguous train / val / test blocks separated by ``gap``.

    Used for sequence data whose rows are overlapping sliding windows (Shakespeare): a random
    split would put near-identical windows in train and test. Dropping ``gap`` rows between
    blocks guarantees that no character is shared across blocks when ``gap >= window length``.
    """
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    train = np.arange(0, max(n_train - gap, 0))
    val = np.arange(n_train, max(n_train + n_val - gap, n_train))
    test = np.arange(n_train + n_val, n)
    return train, val, test
