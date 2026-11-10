"""Ray backend: numerical equivalence with the local executor, timeouts and failure isolation."""

import pytest

pytest.importorskip("ray")

from federated_intelligence.experiments.configuration import ExperimentConfig  # noqa: E402
from federated_intelligence.experiments.runner import run_experiment  # noqa: E402
from tests.conftest import small_experiment  # noqa: E402

pytestmark = [pytest.mark.ray, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _ray_shutdown():
    yield
    import ray

    if ray.is_initialized():
        ray.shutdown()


def test_ray_matches_local_executor_exactly():
    base = small_experiment(federated={"num_rounds": 4, "participation_rate": 0.75})
    local = run_experiment(ExperimentConfig.model_validate(base))
    ray_cfg = small_experiment(federated={"num_rounds": 4, "participation_rate": 0.75},
                               execution={"backend": "ray", "num_workers": 2})
    dist = run_experiment(ExperimentConfig.model_validate(ray_cfg))
    assert dist["summary"]["executor"] == "ray"
    for a, b in zip(local["history"], dist["history"], strict=True):
        assert a.get("participants") == b.get("participants")
        assert a["test_loss"] == pytest.approx(b["test_loss"], abs=1e-5)
        assert a.get("uplink_bytes") == b.get("uplink_bytes")


def test_ray_timeout_yields_timeout_status():
    from federated_intelligence.clients.local_trainer import LocalTrainingConfig
    from federated_intelligence.communication.compression import NoCompression
    from federated_intelligence.experiments.runner import build_dataset, model_kwargs
    from federated_intelligence.models.factory import get_parameters, model_factory
    from federated_intelligence.orchestration.ray_executor import RayExecutor

    cfg = ExperimentConfig.model_validate(small_experiment())
    ds = build_dataset(cfg)
    fn = model_factory(cfg.model.name, init_seed=1, **model_kwargs(cfg, ds))
    ex = RayExecutor(ds.clients, fn, NoCompression(), seed=1, num_workers=2, timeout_s=1e-4)
    try:
        ups = ex.run_round(ds.client_ids[:2], get_parameters(fn()), LocalTrainingConfig(50, 8, 0.01), 1)
        assert {u.status for u in ups} <= {"timeout", "ok"}
        assert any(u.status == "timeout" for u in ups)
    finally:
        ex.shutdown()
