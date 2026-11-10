import copy

import numpy as np

from federated_intelligence.data.validators import check_dataset


def test_clean_dataset_passes(small_dataset):
    rep = check_dataset(small_dataset)
    assert rep.passed, rep.errors
    assert rep.stats["train_val_test_overlap_rows"] == 0
    assert rep.stats["global_test_leakage_rows"] == 0


def test_detects_train_test_leakage(small_dataset):
    ds = copy.deepcopy(small_dataset)
    c = ds.clients["client_000"]
    c.x_test = np.concatenate([c.x_test, c.x_train[:3]])
    c.y_test = np.concatenate([c.y_test, c.y_train[:3]])
    rep = check_dataset(ds)
    assert not rep.passed
    assert any("leakage" in e for e in rep.errors)


def test_detects_global_test_leakage(small_dataset):
    ds = copy.deepcopy(small_dataset)
    c = ds.clients["client_001"]
    c.x_train = np.concatenate([c.x_train, ds.global_test_x[:2]])
    c.y_train = np.concatenate([c.y_train, ds.global_test_y[:2]])
    assert not check_dataset(ds).passed


def test_detects_nan_and_bad_labels(small_dataset):
    ds = copy.deepcopy(small_dataset)
    ds.clients["client_002"].x_train[0, 0] = np.nan
    ds.clients["client_003"].y_val[0] = 7
    rep = check_dataset(ds)
    assert not rep.passed
    assert rep.stats["missing_or_non_finite_values"] == 1
    assert rep.stats["invalid_labels"] == 1


def test_detects_empty_split(small_dataset):
    ds = copy.deepcopy(small_dataset)
    c = ds.clients["client_004"]
    c.x_val, c.y_val = c.x_val[:0], c.y_val[:0]
    assert not check_dataset(ds).passed
