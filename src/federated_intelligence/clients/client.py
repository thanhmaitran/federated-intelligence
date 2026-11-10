"""Federated client: owns local data, never exposes it, returns only (compressed) updates."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from torch import nn

from federated_intelligence.clients.local_trainer import LocalTrainingConfig, train_local
from federated_intelligence.communication.compression import Compressor
from federated_intelligence.data.federated_dataset import ClientSplit
from federated_intelligence.evaluation.metrics import evaluate_model
from federated_intelligence.models.factory import get_parameters, set_parameters
from federated_intelligence.utils.seeds import derive_seed

Params = list[np.ndarray]


@dataclass
class ClientUpdate:
    """Message from a client to the coordinator (the only thing that crosses the boundary)."""

    client_id: str
    status: str  # ok | failed | dropped | rejected | timeout | communication_failure | deadline_exceeded
    num_samples: int = 0
    payload: bytes = b""  # compressed update (delta = w_local - w_global)
    uplink_bytes: int = 0
    train_loss: float = float("nan")
    compute_seconds: float = 0.0
    compression_seconds: float = 0.0
    num_steps: int = 0
    message: str = ""
    extra: dict = field(default_factory=dict)


class FederatedClient:
    """API: get_parameters / set_parameters / train_local / evaluate_local."""

    def __init__(
        self,
        data: ClientSplit,
        model_fn: Callable[[], nn.Module],
        compressor: Compressor,
        experiment_seed: int,
    ):
        self.data = data
        self.client_id = data.client_id
        self._model_fn = model_fn
        self.compressor = compressor
        self.seed = experiment_seed
        self.state: dict = {}  # per-client compressor state (error-feedback residual)
        self._params: Params | None = None

    @property
    def num_train_samples(self) -> int:
        return self.data.num_train

    # ------------------------------------------------------------------ parameters API
    def get_parameters(self) -> Params:
        if self._params is None:
            raise RuntimeError("client has no parameters yet")
        return self._params

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        self._params = [np.array(p, dtype=np.float32, copy=True) for p in parameters]

    # ------------------------------------------------------------------ local training
    def train_local(
        self, global_params: Sequence[np.ndarray], config: LocalTrainingConfig, round_idx: int
    ) -> ClientUpdate:
        """Train from ``global_params`` and return the compressed update delta."""
        # A fresh model per call keeps memory flat: no persistent per-client model copies.
        model = self._model_fn()
        set_parameters(model, global_params)
        seed = derive_seed(self.seed, "train", round_idx, self.client_id)
        res = train_local(model, self.data.x_train, self.data.y_train, config, global_params, seed)
        new_params = get_parameters(model)
        delta = [n - np.asarray(g, dtype=np.float32) for n, g in zip(new_params, global_params, strict=True)]
        t0 = time.perf_counter()
        payload = self.compressor.compress(delta, self.state)
        t_comp = time.perf_counter() - t0
        self.set_parameters(new_params)
        return ClientUpdate(
            client_id=self.client_id,
            status="ok",
            num_samples=res.num_samples,
            payload=payload,
            uplink_bytes=len(payload),
            train_loss=res.train_loss,
            compute_seconds=res.seconds,
            compression_seconds=t_comp,
            num_steps=res.num_steps,
        )

    # ------------------------------------------------------------------ evaluation
    def evaluate_local(
        self, global_params: Sequence[np.ndarray], split: str = "test"
    ) -> dict[str, float]:
        model = self._model_fn()
        set_parameters(model, global_params)
        x, y = getattr(self.data, f"x_{split}"), getattr(self.data, f"y_{split}")
        out = evaluate_model(model, x, y)
        out["num_samples"] = float(len(y))
        return out
