"""Generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from federated_intelligence.data.federated_dataset import FederatedDataset


class BaseGenerator(ABC):
    """A generator turns a configuration plus a seed into a federated dataset."""

    @abstractmethod
    def generate(self) -> FederatedDataset: ...
