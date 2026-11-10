"""Client selection strategies, participation control and dropout simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from federated_intelligence.utils.seeds import derive_seed


@dataclass
class RoundPlan:
    round_idx: int
    selected: list[str]
    dropped: list[str]  # selected but dropped out before sending an update (simulated)

    @property
    def participants(self) -> list[str]:
        gone = set(self.dropped)
        return [c for c in self.selected if c not in gone]


class ClientSelector(ABC):
    name = "base"

    def __init__(self, client_ids: list[str], participation_rate: float, seed: int):
        if not 0 < participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        self.client_ids = sorted(client_ids)
        self.rate = participation_rate
        self.seed = seed

    def _rng(self, round_idx: int) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, "select", self.name, round_idx))

    @abstractmethod
    def select(self, round_idx: int) -> list[str]: ...


class RandomSelector(ClientSelector):
    """Uniformly random subset of fixed size max(1, round(rate * K)) without replacement."""

    name = "random"

    def select(self, round_idx: int) -> list[str]:
        k = max(1, round(self.rate * len(self.client_ids)))
        rng = self._rng(round_idx)
        idx = rng.choice(len(self.client_ids), size=k, replace=False)
        return [self.client_ids[i] for i in sorted(idx)]


class PoissonSelector(ClientSelector):
    """Each client is included independently with probability ``rate`` (needed for DP accounting).

    May return an empty list.
    """

    name = "poisson"

    def select(self, round_idx: int) -> list[str]:
        rng = self._rng(round_idx)
        mask = rng.random(len(self.client_ids)) < self.rate
        return [c for c, m in zip(self.client_ids, mask, strict=True) if m]


class AvailabilitySelector(ClientSelector):
    """Only currently available clients are eligible; then sample ``rate`` of *all* K clients.

    ``is_available(client_id, round_idx)`` typically comes from the network simulator.
    """

    name = "availability"

    def __init__(
        self,
        client_ids: list[str],
        participation_rate: float,
        seed: int,
        is_available: Callable[[str, int], bool],
    ):
        super().__init__(client_ids, participation_rate, seed)
        self.is_available = is_available

    def select(self, round_idx: int) -> list[str]:
        eligible = [c for c in self.client_ids if self.is_available(c, round_idx)]
        if not eligible:
            return []
        k = min(len(eligible), max(1, round(self.rate * len(self.client_ids))))
        rng = self._rng(round_idx)
        idx = rng.choice(len(eligible), size=k, replace=False)
        return [eligible[i] for i in sorted(idx)]


def simulate_dropout(
    selected: list[str], dropout_rate: float, seed: int, round_idx: int
) -> list[str]:
    """Each selected client independently drops out with probability ``dropout_rate``."""
    if dropout_rate <= 0:
        return []
    out = []
    for c in selected:
        rng = np.random.default_rng(derive_seed(seed, "dropout", round_idx, c))
        if rng.random() < dropout_rate:
            out.append(c)
    return out


def build_selector(
    kind: str,
    client_ids: list[str],
    rate: float,
    seed: int,
    is_available: Callable[[str, int], bool] | None = None,
) -> ClientSelector:
    if kind == "random":
        return RandomSelector(client_ids, rate, seed)
    if kind == "poisson":
        return PoissonSelector(client_ids, rate, seed)
    if kind == "availability":
        if is_available is None:
            is_available = lambda c, r: True  # noqa: E731
        return AvailabilitySelector(client_ids, rate, seed, is_available)
    raise ValueError(f"Unknown selection strategy '{kind}'")


class RoundManager:
    """Plans a round: selection followed by dropout."""

    def __init__(self, selector: ClientSelector, dropout_rate: float, seed: int):
        self.selector = selector
        self.dropout_rate = dropout_rate
        self.seed = seed

    def plan(self, round_idx: int) -> RoundPlan:
        selected = self.selector.select(round_idx)
        dropped = simulate_dropout(selected, self.dropout_rate, self.seed, round_idx)
        return RoundPlan(round_idx, selected, dropped)
