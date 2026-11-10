from __future__ import annotations

import numpy as np
import pytest

from federated_intelligence.data.generators import GeneratorConfig, Scenario, generate_dataset


@pytest.fixture(scope="session")
def small_cfg() -> GeneratorConfig:
    return GeneratorConfig(
        seed=7,
        num_clients=8,
        num_features=10,
        min_samples_per_client=100,
        max_samples_per_client=300,
        global_test_size=800,
        scenario=Scenario.COMBINED,
    )


@pytest.fixture(scope="session")
def small_dataset(small_cfg):
    return generate_dataset(small_cfg).standardized()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def small_experiment(**overrides) -> dict:
    """A tiny, fast experiment config (dict) used by integration and system tests."""
    cfg = {
        "experiment": {"name": "test", "seed": 3},
        "dataset": {
            "scenario": "label_skew",
            "num_clients": 8,
            "num_features": 10,
            "min_samples_per_client": 80,
            "max_samples_per_client": 160,
            "global_test_size": 600,
        },
        "model": {"name": "logreg"},
        "federated": {"num_rounds": 4, "participation_rate": 0.5},
        "training": {"local_epochs": 1, "batch_size": 32, "learning_rate": 0.1},
    }
    for k, v in overrides.items():
        cfg.setdefault(k, {}).update(v)
    return cfg
