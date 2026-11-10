"""Pydantic response/request schemas of the REST API."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict


class Health(BaseModel):
    status: str
    version: str
    database: str


class DatasetInfo(BaseModel):
    name: str
    kind: str  # synthetic | natural
    description: str
    status: str  # available | not_downloaded | generated | not_generated
    num_clients: int | None = None
    metadata: dict[str, Any] = {}


class PartitionInfo(BaseModel):
    partition_id: str
    dataset: str
    method: str
    alpha: float | None = None
    num_clients: int
    total_samples: int


class ClientInfo(BaseModel):
    client_id: str
    num_samples: int | None = None
    train_samples: int | None = None
    validation_samples: int | None = None
    test_samples: int | None = None
    metrics: dict[str, float] = {}


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    exp_group: str
    variant: str
    seed: int
    status: str
    source: str
    algorithm: str | None = None
    test_accuracy: float | None = None
    total_bytes: int | None = None
    epsilon: float | None = None
    created_at: dt.datetime
    finished_at: dt.datetime | None = None


class ExperimentDetail(ExperimentSummary):
    config: dict[str, Any]
    summary: dict[str, Any]
    error: str = ""


class ExperimentCreate(BaseModel):
    """Run configuration; unspecified sections take the library defaults."""

    config: dict[str, Any] = {}
    group: str = "API"


class RoundOut(BaseModel):
    round: int
    metrics: dict[str, Any]


class ComparisonRow(BaseModel):
    key: dict[str, Any]
    n: int
    mean: float
    sd: float
    ci_low: float
    ci_high: float


class Overview(BaseModel):
    num_experiments: int
    by_status: dict[str, int]
    by_group: dict[str, int]
    best_test_accuracy: float | None
    total_bytes_transferred: int
    datasets: int
