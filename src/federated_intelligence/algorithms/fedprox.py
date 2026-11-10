"""FedProx (Li et al., 2020).

Server aggregation is identical to FedAvg; the difference lives in the local objective

    L_prox(w) = L(w) + (mu / 2) ||w - w_t||_2^2

which is applied by ``clients.local_trainer`` through ``LocalTrainingConfig.mu``. With mu = 0
FedProx reduces exactly to FedAvg.
"""

from __future__ import annotations

from federated_intelligence.algorithms.fedavg import FedAvg


class FedProx(FedAvg):
    name = "fedprox"

    def __init__(self, mu: float = 0.01):
        if mu < 0:
            raise ValueError("mu must be >= 0")
        self.mu = mu


def build_algorithm(name: str, mu: float = 0.0) -> FedAvg:
    if name == "fedavg":
        return FedAvg()
    if name == "fedprox":
        return FedProx(mu)
    raise ValueError(f"Unknown algorithm '{name}'")
