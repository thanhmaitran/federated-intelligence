"""FastAPI backend (SQLite in-memory): registry, creation, metrics, analysis."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from federated_intelligence.api.importer import store_run  # noqa: E402
from federated_intelligence.api.main import create_app  # noqa: E402
from federated_intelligence.experiments.configuration import ExperimentConfig  # noqa: E402
from federated_intelligence.experiments.runner import run_experiment  # noqa: E402
from tests.conftest import small_experiment  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    meta = tmp_path_factory.mktemp("data") / "metadata"
    meta.mkdir()
    (meta / "partition_demo.json").write_text(json.dumps({
        "partition_id": "demo", "dataset": "synthetic_federated_tabular", "partition_method": "dirichlet", "alpha": 0.5,
        "num_clients": 1, "total_samples": 10,
        "clients": {"client_000": {"num_samples": 10, "train_samples": 7, "validation_samples": 1, "test_samples": 2}}}))
    app = create_app("sqlite:///:memory:", data_dir=meta.parent, run_inline=True)
    with TestClient(app) as c:
        # two finished runs of one group, as the importer would load them
        for seed in (1, 2):
            res = run_experiment(ExperimentConfig.model_validate(small_experiment(experiment={"seed": seed, "name": "a"})))
            res["exp_id"], res["variant"], res["axes"] = "EXP-T", "a", {}
            with app.state.session_factory() as s:
                store_run(s, res)
                s.commit()
        yield c


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["database"] == "sqlite"


def test_datasets_and_partitions_and_clients(client):
    names = {d["name"] for d in client.get("/api/datasets").json()}
    assert {"synthetic/iid", "femnist", "shakespeare"} <= names
    assert client.get("/api/partitions").json()[0]["partition_id"] == "demo"
    assert client.get("/api/clients", params={"partition_id": "demo"}).json()[0]["train_samples"] == 7
    assert client.get("/api/clients", params={"partition_id": "nope"}).status_code == 404


def test_list_detail_metrics_rounds(client):
    rows = client.get("/api/experiments", params={"group": "EXP-T"}).json()
    assert len(rows) == 2 and rows[0]["status"] == "finished" and rows[0]["test_accuracy"] > 0.5
    eid = rows[0]["id"]
    d = client.get(f"/api/experiments/{eid}").json()
    assert d["config"]["federated"]["algorithm"] == "fedavg"
    m = client.get(f"/api/experiments/{eid}/metrics").json()["metrics"]
    assert len(m["test_accuracy"]) == 5  # rounds 0..4
    assert len(client.get(f"/api/experiments/{eid}/rounds").json()) == 5
    assert client.get("/api/experiments/EXP-T/zzz").status_code == 404
    per = client.get("/api/clients", params={"experiment_id": eid}).json()
    assert len(per) == 8 and "accuracy" in per[0]["metrics"]


def test_create_experiment_runs_and_validates(client):
    body = {"group": "API", "config": small_experiment(experiment={"name": "via_api"})}
    r = client.post("/api/experiments", json=body)
    assert r.status_code == 202
    created = client.get("/api/experiments", params={"group": "API"}).json()
    assert created and created[0]["status"] == "finished" and created[0]["source"] == "api"
    bad = client.post("/api/experiments", json={"config": {"federated": {"algorithm": "fedavg", "mu": 1.0}}})
    assert bad.status_code == 422


def test_compare_and_overview(client):
    rows = client.get("/api/analysis/compare", params={"group": "EXP-T", "metric": "test_accuracy"}).json()
    assert rows[0]["n"] == 2 and rows[0]["ci_low"] <= rows[0]["mean"] <= rows[0]["ci_high"]
    assert client.get("/api/analysis/compare", params={"group": "EXP-T", "metric": "bogus"}).status_code == 422
    ov = client.get("/api/overview").json()
    assert ov["num_experiments"] >= 3 and ov["total_bytes_transferred"] > 0
