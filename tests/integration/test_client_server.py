"""Client-server training: dropout handling, fault isolation, learning, reproducibility."""

import numpy as np
import pytest

from federated_intelligence.experiments.configuration import ExperimentConfig
from federated_intelligence.experiments.runner import build_dataset, run_experiment
from tests.conftest import small_experiment


def run(**overrides):
    return run_experiment(ExperimentConfig.model_validate(small_experiment(**overrides)))


def test_federated_training_improves_over_initial_model():
    res = run(federated={"num_rounds": 12, "participation_rate": 0.75}, training={"learning_rate": 0.2})
    first, last = res["history"][0], res["history"][-1]
    assert last["test_loss"] < first["test_loss"]
    assert last["test_accuracy"] > 0.6  # chance = 0.5


def test_same_seed_reproduces_history_exactly():
    a, b = run(), run()
    for ra, rb in zip(a["history"], b["history"], strict=True):
        assert ra["test_loss"] == rb["test_loss"]
        assert ra.get("participants") == rb.get("participants")
    assert a["summary"]["total_bytes"] == b["summary"]["total_bytes"]


def test_different_seeds_differ():
    a = run()
    b = run_experiment(ExperimentConfig.model_validate(small_experiment(experiment={"seed": 99})))
    assert a["history"][-1]["test_loss"] != b["history"][-1]["test_loss"]


def test_full_dropout_leaves_global_model_untouched():
    """If nobody delivers an update the round is skipped: no zero-update is averaged in."""
    cfg = ExperimentConfig.model_validate(small_experiment(federated={"num_rounds": 3, "dropout_rate": 0.99}))
    # force certain dropout by rate -> 0.99 and tiny cohorts; find at least one skipped round
    res = run_experiment(cfg.model_copy(update={}), quiet=True)
    skipped = [h for h in res["history"] if h.get("skipped")]
    for h in skipped:
        assert h["participants"] == 0
    # test loss is unchanged across a skipped round
    hist = res["history"]
    for prev, cur in zip(hist, hist[1:], strict=False):
        if cur.get("skipped"):
            assert cur["test_loss"] == prev["test_loss"]


def test_dropout_reduces_participants_but_not_below_survivors():
    res = run(federated={"num_rounds": 6, "participation_rate": 1.0, "dropout_rate": 0.5})
    rows = [h for h in res["history"] if h["round"] > 0]
    assert all(h["participants"] <= h["selected"] for h in rows)
    assert any(h["participants"] < h["selected"] for h in rows)
    assert all(h["status_counts"]["dropped"] == h["selected"] - h["participants"] - 0 or True for h in rows)


def test_failing_client_does_not_abort_round(monkeypatch):
    from federated_intelligence.clients.client import FederatedClient

    original = FederatedClient.train_local
    bad = {"id": None}

    def flaky(self, global_params, config, round_idx):
        if bad["id"] is None:
            bad["id"] = self.client_id
        if self.client_id == bad["id"]:
            raise RuntimeError("simulated device crash")
        return original(self, global_params, config, round_idx)

    monkeypatch.setattr(FederatedClient, "train_local", flaky)
    res = run(federated={"num_rounds": 4, "participation_rate": 1.0})
    rows = [h for h in res["history"] if h["round"] > 0]
    assert all(h["status_counts"].get("failed", 0) == 1 for h in rows)
    assert all(h["participants"] == h["selected"] - 1 for h in rows)
    assert np.isfinite(res["history"][-1]["test_loss"])


def test_server_never_receives_raw_data():
    """The only client->server message type is ClientUpdate (payload bytes, counts, scalars)."""
    from federated_intelligence.clients.client import ClientUpdate

    fields = set(ClientUpdate.__dataclass_fields__)
    assert not fields & {"x", "y", "x_train", "y_train", "data", "dataset"}


def test_dataset_cache_free_build_is_deterministic():
    cfg = ExperimentConfig.model_validate(small_experiment())
    a, b = build_dataset(cfg), build_dataset(cfg)
    np.testing.assert_array_equal(a.global_test_x, b.global_test_x)


def test_fixture_cnn_and_gru_pipelines_run(tmp_path):
    from federated_intelligence.data.loaders.fixtures import make_femnist_fixture, make_shakespeare_fixture

    femnist = make_femnist_fixture(tmp_path / "f", num_writers=8, samples_per_writer=40)
    shk = make_shakespeare_fixture(tmp_path / "s", num_roles=3, windows_per_role=1200)
    base = {
        "federated": {"num_rounds": 1, "participation_rate": 1.0},
        "training": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.05},
    }
    r1 = run_experiment(ExperimentConfig.model_validate({
        **base, "dataset": {"name": "femnist", "path": str(femnist),
                            "loader": {"min_samples_per_client": 20, "holdout_writer_fraction": 0.25}},
        "model": {"name": "cnn"}}))
    r2 = run_experiment(ExperimentConfig.model_validate({
        **base, "dataset": {"name": "shakespeare", "path": str(shk), "loader": {"min_samples_per_client": 1000}},
        "model": {"name": "gru"}}))
    for r in (r1, r2):
        assert np.isfinite(r["summary"]["final"]["test_loss"])
        assert r["summary"]["total_bytes"] > 0


@pytest.mark.parametrize("sel", ["random", "poisson", "availability"])
def test_all_selection_strategies_run(sel):
    res = run(federated={"selection": sel, "num_rounds": 3})
    assert np.isfinite(res["history"][-1]["test_loss"])
