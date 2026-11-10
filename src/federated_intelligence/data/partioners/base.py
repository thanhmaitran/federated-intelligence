"""Partitioner interface: map sample indices of a pooled dataset to clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

PARTITIONER_VERSION = "1.0.0"

Partition = dict[str, np.ndarray]


class BasePartitioner(ABC):
    method: str = "base"

    def __init__(self, num_clients: int, seed: int = 42):
        if num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        self.num_clients = num_clients
        self.seed = seed

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @staticmethod
    def client_name(k: int) -> str:
        return f"client_{k:03d}"

    @abstractmethod
    def partition(self, labels: np.ndarray) -> Partition:
        """Return {client_id: sorted sample indices}. Every index is used exactly once."""

    def params(self) -> dict:
        return {}

    def _to_partition(self, buckets: list[list[int]] | list[np.ndarray]) -> Partition:
        return {
            self.client_name(k): np.sort(np.asarray(b, dtype=np.int64))
            for k, b in enumerate(buckets)
        }
