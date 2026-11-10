"""Coarse performance guards: runtime, memory and communication budgets (generous thresholds)."""

import resource
import time

import numpy as np
import pytest

from federated_intelligence.experiments.configuration import ExperimentConfig
from federated_intelligence.experiments.runner import run_experiment
from tests.conftest import small_experiment

pytestmark = pytest.mark.performance


def test_round_runtime_budget():
    cfg = ExperimentConfig.model_validate(small_experiment(federated={"num_rounds": 10, "participation_rate": 1.0}))
    t0 = time.perf_counter()
    run_experiment(cfg)
    per_round = (time.perf_counter() - t0) / 10
    assert per_round < 1.5, f"round took {per_round:.2f}s on an 8-client logistic-regression toy problem"


def test_peak_memory_budget():
    run_experiment(ExperimentConfig.model_validate(small_experiment()))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB
    assert peak_mb < 3000


def test_communication_is_exactly_accounted():
    """Uplink bytes = participants x measured payload size (float32 baseline)."""
    res = run_experiment(ExperimentConfig.model_validate(small_experiment(federated={"participation_rate": 1.0})))
    s = res["summary"]
    assert s["total_uplink_bytes"] == s["baseline_float32_update_bytes"] * s["total_participating_updates"]
    assert s["uplink_compression_ratio"] == pytest.approx(1.0)


def test_int8_and_topk_shrink_real_payloads():
    base = run_experiment(ExperimentConfig.model_validate(small_experiment(model={"name": "mlp"}))).get("summary")
    int8 = run_experiment(ExperimentConfig.model_validate(
        small_experiment(model={"name": "mlp"}, communication={"compressor": "int8"})))["summary"]
    topk = run_experiment(ExperimentConfig.model_validate(
        small_experiment(model={"name": "mlp"}, communication={"compressor": "topk", "topk_ratio": 0.05})))["summary"]
    assert int8["mean_uplink_bytes_per_update"] < 0.35 * base["mean_uplink_bytes_per_update"]
    assert topk["mean_uplink_bytes_per_update"] < 0.5 * base["mean_uplink_bytes_per_update"]
    assert np.isfinite(topk["final"]["test_loss"])
