"""Ray backend: clients live inside Ray actors; the coordinator only exchanges updates.

Client datasets are placed into the worker actors once at start-up (simulating on-device data);
afterwards only global parameters go down and compressed updates come back. Every task has a
timeout and failures are converted into ``ClientUpdate(status='failed' | 'timeout')`` so a broken
client never aborts the round.

Per-(round, client) seeds are derived from the experiment seed, so results do not depend on which
worker runs a client - the Ray backend is numerically equivalent to ``LocalExecutor``.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Sequence

import numpy as np

from federated_intelligence.clients.client import ClientUpdate, FederatedClient
from federated_intelligence.clients.local_trainer import LocalTrainingConfig
from federated_intelligence.communication.compression import Compressor
from federated_intelligence.data.federated_dataset import ClientSplit


def _require_ray():
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised only without Ray installed
        raise RuntimeError(
            "Ray is not installed. Install the optional extra: pip install '.[distributed]'"
        ) from exc
    return ray


class _WorkerImpl:
    """Holds a subset of clients inside one Ray actor process."""

    def __init__(self, splits: list[ClientSplit], model_fn, compressor: Compressor, seed: int):
        import torch

        torch.set_num_threads(1)
        self.clients = {s.client_id: FederatedClient(s, model_fn, compressor, seed) for s in splits}

    def train(self, client_id, global_params, config, round_idx) -> ClientUpdate:
        try:
            return self.clients[client_id].train_local(global_params, config, round_idx)
        except Exception as exc:
            return ClientUpdate(client_id, "failed", message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}")

    def evaluate(self, global_params, split):
        return {cid: c.evaluate_local(global_params, split) for cid, c in self.clients.items()}


class RayExecutor:
    name = "ray"

    def __init__(
        self,
        splits: dict[str, ClientSplit],
        model_fn: Callable,
        compressor: Compressor,
        seed: int,
        num_workers: int = 2,
        timeout_s: float = 120.0,
    ):
        self.ray = _require_ray()
        if not self.ray.is_initialized():
            self.ray.init(num_cpus=num_workers, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True)
        self.timeout_s = timeout_s
        ids = sorted(splits)
        num_workers = max(1, min(num_workers, len(ids)))
        shards: list[list[ClientSplit]] = [[] for _ in range(num_workers)]
        self.owner: dict[str, int] = {}
        for i, cid in enumerate(ids):  # round-robin placement
            shards[i % num_workers].append(splits[cid])
            self.owner[cid] = i % num_workers
        actor_cls = self.ray.remote(num_cpus=1)(_WorkerImpl)
        self.workers = [actor_cls.remote(shard, model_fn, compressor, seed) for shard in shards]

    def run_round(
        self,
        client_ids: Sequence[str],
        global_params: Sequence[np.ndarray],
        config: LocalTrainingConfig,
        round_idx: int,
    ) -> list[ClientUpdate]:
        params_ref = self.ray.put(list(global_params))  # one copy in the object store
        refs = {cid: self.workers[self.owner[cid]].train.remote(cid, params_ref, config, round_idx) for cid in client_ids}
        updates: dict[str, ClientUpdate] = {}
        pending = dict(refs)
        ready, not_ready = self.ray.wait(list(pending.values()), num_returns=len(pending), timeout=self.timeout_s)
        ready_set = set(ready)
        for cid, ref in pending.items():
            if ref in ready_set:
                try:
                    updates[cid] = self.ray.get(ref)
                except Exception as exc:
                    updates[cid] = ClientUpdate(cid, "failed", message=f"{type(exc).__name__}: {exc}")
            else:
                self.ray.cancel(ref, force=False)
                updates[cid] = ClientUpdate(cid, "timeout", message=f"no result within {self.timeout_s}s")
        return [updates[c] for c in client_ids]

    def evaluate_clients(self, global_params: Sequence[np.ndarray], split: str = "test") -> dict[str, dict[str, float]]:
        params_ref = self.ray.put(list(global_params))
        parts = self.ray.get([w.evaluate.remote(params_ref, split) for w in self.workers])
        merged: dict[str, dict[str, float]] = {}
        for p in parts:
            merged.update(p)
        return dict(sorted(merged.items()))

    def shutdown(self) -> None:
        for w in self.workers:
            self.ray.kill(w)
        self.workers = []
