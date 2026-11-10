import numpy as np
import pandas as pd
import pytest

from federated_intelligence.data.generators import (
    GeneratorConfig,
    Scenario,
    TabularGenerator,
    generate_dataset,
)
from federated_intelligence.data.profiling import heterogeneity_summary


def cfg(scenario, **kw):
    base = dict(
        seed=11,
        num_clients=12,
        num_features=8,
        min_samples_per_client=100,
        max_samples_per_client=400,
        global_test_size=600,
        scenario=scenario,
    )
    base.update(kw)
    return GeneratorConfig(**base)


def test_same_seed_is_bit_identical():
    a = generate_dataset(cfg(Scenario.COMBINED))
    b = generate_dataset(cfg(Scenario.COMBINED))
    for cid in a.client_ids:
        np.testing.assert_array_equal(a.clients[cid].x_train, b.clients[cid].x_train)
        np.testing.assert_array_equal(a.clients[cid].y_test, b.clients[cid].y_test)
    np.testing.assert_array_equal(a.global_test_x, b.global_test_x)


def test_different_seed_differs():
    a = generate_dataset(cfg(Scenario.IID))
    b = generate_dataset(cfg(Scenario.IID, seed=12))
    assert not np.allclose(a.clients["client_000"].x_train[:5], b.clients["client_000"].x_train[:5])


def test_split_fractions_and_totals():
    ds = generate_dataset(cfg(Scenario.IID))
    for c in ds.clients.values():
        n = c.num_samples
        assert abs(c.num_train / n - 0.70) < 0.02
        assert len(c.y_val) > 0 and len(c.y_test) > 0


def test_scenarios_isolate_their_heterogeneity_source():
    h = {s: heterogeneity_summary(generate_dataset(cfg(s))) for s in Scenario}
    iid = h[Scenario.IID]
    assert h[Scenario.LABEL_SKEW]["mean_pairwise_label_jsd"] > 10 * iid["mean_pairwise_label_jsd"]
    assert h[Scenario.QUANTITY_SKEW]["sample_count_cv"] > 3 * iid["sample_count_cv"] + 0.1
    assert h[Scenario.FEATURE_SKEW]["feature_mean_distance"] > 1.5 * iid["feature_mean_distance"]
    # label skew must not alter quantity or (much) features
    assert h[Scenario.LABEL_SKEW]["sample_count_cv"] < 0.05


def test_calibrated_positive_rate_matches_target():
    gen = TabularGenerator(cfg(Scenario.LABEL_SKEW, num_clients=6, min_samples_per_client=3000,
                               max_samples_per_client=3000))
    ds = gen.generate()
    for p in gen.client_params:
        c = ds.clients[p.client_id]
        observed = np.concatenate([c.y_train, c.y_val, c.y_test]).mean()
        assert abs(observed - p.positive_rate) < 0.04


def test_global_test_not_taken_from_client_splits():
    ds = generate_dataset(cfg(Scenario.IID))
    client_rows = {r.tobytes() for c in ds.clients.values() for r in c.x_test}
    assert not any(r.tobytes() in client_rows for r in ds.global_test_x)


def test_standardizer_uses_train_statistics_only(small_cfg):
    raw = generate_dataset(small_cfg)
    std = raw.standardized()
    x_tr, _ = std.pooled_train()
    np.testing.assert_allclose(x_tr.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(x_tr.std(axis=0), 1, atol=1e-3)
    # test data is transformed with the train scaler, so it is generally NOT exactly standardized
    xt = np.concatenate([c.x_test for c in std.clients.values()])
    assert not np.allclose(xt.mean(axis=0), 0, atol=1e-6)


def test_write_parquet_and_manifest(tmp_path, small_cfg):
    gen = TabularGenerator(small_cfg)
    manifest = gen.write(tmp_path)
    assert manifest["seed"] == small_cfg.seed
    assert manifest["scenario"] == "combined_non_iid"
    df = pd.read_parquet(tmp_path / "clients.parquet")
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["client_id"].nunique() == small_cfg.num_clients
    for name, meta in manifest["files"].items():
        assert len(meta["sha256"]) == 64 and (tmp_path / name).exists()


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        GeneratorConfig(min_samples_per_client=500, max_samples_per_client=100)
    with pytest.raises(ValueError):
        GeneratorConfig(train_fraction=0.5, val_fraction=0.2, test_fraction=0.2)
