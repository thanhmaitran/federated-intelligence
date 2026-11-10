"""Typed, YAML-driven experiment configuration (no hyper-parameters are hard-coded in training)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentMeta(BaseModel):
    name: str = "experiment"
    seed: int = 42
    description: str = ""


class DatasetConfig(BaseModel):
    """Synthetic federated tabular dataset.

    ``mode='native'``: per-client generator with a heterogeneity ``scenario``.
    ``mode='pooled'``: sample one population and partition it (``partition`` + ``alpha``).
    """

    name: Literal["synthetic_federated_tabular", "femnist", "shakespeare"] = "synthetic_federated_tabular"
    # femnist / shakespeare: where ``scripts/download_datasets.py`` saved the Arrow dataset, and
    # keyword arguments forwarded to the loader (max_clients, fractions, holdout_writer_fraction, ...).
    path: str | None = None
    loader: dict[str, Any] = {}
    mode: Literal["native", "pooled"] = "native"
    scenario: str = "iid"
    partition: Literal["iid", "dirichlet", "quantity_skew"] = "iid"
    alpha: float = 0.5
    total_samples: int = 20000
    num_clients: int = 30
    num_features: int = 24
    min_samples_per_client: int = 300
    max_samples_per_client: int = 2000
    global_test_size: int = 5000
    label_beta: float = 0.5
    feature_shift: float = 1.0
    quantity_sigma: float = 1.0
    concept_noise: float = 0.05
    label_noise: float = 0.0
    weight_scale: float = 2.0  # signal strength of the true labelling function


class ModelConfig(BaseModel):
    name: Literal["logreg", "mlp", "cnn", "gru"] = "mlp"
    hidden: list[int] = [64, 32]


class FederatedConfig(BaseModel):
    algorithm: Literal["fedavg", "fedprox"] = "fedavg"
    mu: float = Field(0.0, ge=0)
    participation_rate: float = Field(0.5, gt=0, le=1)
    num_rounds: int = Field(50, ge=1)
    selection: Literal["random", "poisson", "availability"] = "random"
    dropout_rate: float = Field(0.0, ge=0, lt=1)
    server_lr: float = Field(1.0, gt=0)
    eval_every: int = Field(1, ge=1)


class TrainingConfig(BaseModel):
    local_epochs: int = Field(3, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(0.05, gt=0)
    momentum: float = 0.0
    weight_decay: float = 0.0


class PrivacyConfig(BaseModel):
    enabled: bool = False
    clip_norm: float = Field(1.0, gt=0)
    noise_multiplier: float | None = Field(1.0, ge=0)
    target_epsilon: float | None = None  # if set, sigma is derived with the accountant
    delta: float = Field(1e-5, gt=0, lt=1)


class CommunicationConfig(BaseModel):
    compressor: Literal["none", "float16", "int8", "topk"] = "none"
    topk_ratio: float = Field(0.1, gt=0, le=1)
    error_feedback: bool = False
    downlink: Literal["float32", "float16"] = "float32"


class NetworkConfig(BaseModel):
    enabled: bool = False
    latency_ms: float = 50.0
    bandwidth_mbps: float = 10.0
    heterogeneity: float = 0.5
    failure_prob: float = 0.0
    availability: float = 1.0
    deadline_s: float | None = None


class ExecutionConfig(BaseModel):
    backend: Literal["local", "ray"] = "local"
    num_workers: int = 2
    client_timeout_s: float = 120.0


class EvaluationConfig(BaseModel):
    metrics: list[str] = ["accuracy", "macro_f1", "loss"]
    per_client: bool = True
    per_client_every: int = Field(10, ge=1)  # per-client evaluation is the costly part
    target_accuracy: float = 0.80


class ExperimentConfig(BaseModel):
    experiment: ExperimentMeta = ExperimentMeta()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    federated: FederatedConfig = FederatedConfig()
    training: TrainingConfig = TrainingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    communication: CommunicationConfig = CommunicationConfig()
    network: NetworkConfig = NetworkConfig()
    execution: ExecutionConfig = ExecutionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def _consistency(self) -> ExperimentConfig:
        if self.federated.algorithm == "fedavg" and self.federated.mu != 0:
            raise ValueError("FedAvg requires mu = 0; use algorithm 'fedprox' for mu > 0")
        if self.privacy.enabled and self.federated.selection == "availability":
            raise ValueError("DP accounting assumes Poisson sampling; availability-based selection is not allowed")
        return self

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        _deep_update(raw, overrides)
    return ExperimentConfig.model_validate(raw)


def config_from_dict(raw: dict, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw = json.loads(json.dumps(raw))
    if overrides:
        _deep_update(raw, overrides)
    return ExperimentConfig.model_validate(raw)
