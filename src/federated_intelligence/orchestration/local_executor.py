"""In-process sequential executor: the reference backend (no extra dependencies)."""

from __future__ import annotations

import traceback
from collections.abc import Sequence

import numpy as np
import torch

from federated_intelligence.clients.client import ClientUpdate, FederatedClient
from federated_intelligence.clients.local_trainer import LocalTrainingConfig


class LocalExecutor:
    name = "local"

    def __init__(self, clients: dict[str, FederatedClient]):
        torch.set_num_threads(1)
        self.clients = clients

    def run_round(
        self,
        client_ids: Sequence[str],
        global_params: Sequence[np.ndarray],
        config: LocalTrainingConfig,
        round_idx: int,
    ) -> list[ClientUpdate]:
        updates = []
        for cid in client_ids:
            try:
                updates.append(self.clients[cid].train_local(global_params, config, round_idx))
            except Exception as exc:  # fault isolation: one failing client never kills the round
                updates.append(
                    ClientUpdate(cid, "failed", message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}")
                )
        return updates

    def evaluate_clients(
        self, global_params: Sequence[np.ndarray], split: str = "test"
    ) -> dict[str, dict[str, float]]:
        return {cid: c.evaluate_local(global_params, split) for cid, c in sorted(self.clients.items())}

    def shutdown(self) -> None:
        pass
