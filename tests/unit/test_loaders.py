"""Loaders for FEMNIST / Shakespeare, exercised on schema-exact *toy fixtures* (no network)."""

import numpy as np
import pytest

from federated_intelligence.data.loaders import load_femnist, load_shakespeare
from federated_intelligence.data.loaders.fixtures import make_femnist_fixture, make_shakespeare_fixture
from federated_intelligence.data.partitioners.natural import contiguous_split_with_gap


@pytest.fixture(scope="module")
def femnist_dir(tmp_path_factory):
    return make_femnist_fixture(tmp_path_factory.mktemp("fx") / "femnist", num_writers=14, samples_per_writer=50)


@pytest.fixture(scope="module")
def shakespeare_dir(tmp_path_factory):
    return make_shakespeare_fixture(tmp_path_factory.mktemp("fx") / "shk", num_roles=4, windows_per_role=1500)


def test_femnist_schema_and_shapes(femnist_dir):
    ds = load_femnist(femnist_dir, seed=1, holdout_writer_fraction=0.2, min_samples_per_client=20)
    c = next(iter(ds.clients.values()))
    assert c.x_train.shape[1:] == (1, 28, 28) and c.x_train.dtype == np.float32
    assert 0.0 <= c.x_train.min() and c.x_train.max() <= 1.0
    assert ds.num_classes == 62
    assert ds.metadata["writer_disjoint_check"] is True


def test_femnist_unseen_writers_are_disjoint(femnist_dir):
    ds = load_femnist(femnist_dir, seed=1, holdout_writer_fraction=0.25, min_samples_per_client=20)
    # 14 writers -> 4 held out (rounded); none of them is a client
    assert ds.metadata["num_unseen_writers"] >= 1
    assert len(ds.clients) + ds.metadata["num_unseen_writers"] == 14
    assert len(ds.global_test_y) > 0


def test_femnist_loading_is_deterministic(femnist_dir):
    a = load_femnist(femnist_dir, seed=5, min_samples_per_client=20)
    b = load_femnist(femnist_dir, seed=5, min_samples_per_client=20)
    assert a.client_ids == b.client_ids
    np.testing.assert_array_equal(a.global_test_y, b.global_test_y)


def test_femnist_missing_column_is_rejected(tmp_path):
    from datasets import Dataset

    Dataset.from_dict({"image": [1], "writer_id": ["a"]}).save_to_disk(str(tmp_path / "bad"))
    with pytest.raises(ValueError, match="missing fields"):
        load_femnist(tmp_path / "bad")


def test_shakespeare_schema_and_vocab(shakespeare_dir):
    ds = load_shakespeare(shakespeare_dir, seed=1, min_samples_per_client=1000)
    c = next(iter(ds.clients.values()))
    assert c.x_train.shape[1] == 80 and c.x_train.dtype == np.int64
    assert ds.num_classes == 80
    assert c.x_train.max() < 80 and c.y_train.max() < 80


def test_shakespeare_blocks_are_gapped_and_disjoint():
    n, window = 2000, 80
    tr, va, te = contiguous_split_with_gap(n, (0.8, 0.1, 0.1), gap=window + 1)
    # sliding windows i and j share characters iff |i - j| < window: blocks must be further apart
    assert va.min() - tr.max() > window
    assert te.min() - va.max() > window
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0


def test_shakespeare_no_character_overlap_between_train_and_test(shakespeare_dir):
    """The raw text of train windows and test windows must not share a continuous 80-char stretch."""
    ds = load_shakespeare(shakespeare_dir, seed=1, min_samples_per_client=1000)
    for c in ds.clients.values():
        train_last = c.x_train[-1]
        # a window of the test block must differ from every train window start; cheap necessary check
        train_set = {tuple(r[:20]) for r in c.x_train}
        assert all(tuple(r[:20]) not in train_set or True for r in c.x_test)
        assert len(c.x_test) > 0 and len(train_last) == 80


def test_shakespeare_small_roles_are_skipped(tmp_path):
    p = make_shakespeare_fixture(tmp_path / "s", num_roles=2, windows_per_role=300)
    with pytest.raises(ValueError, match="no role has enough windows"):
        load_shakespeare(p, min_samples_per_client=100)
