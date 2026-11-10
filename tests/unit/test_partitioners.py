import numpy as np
import pytest

from federated_intelligence.data.partitioners import (
    DirichletPartitioner,
    IIDPartitioner,
    NaturalPartitioner,
    QuantitySkewPartitioner,
    build_manifest,
    partition_checksum,
    split_groups,
)
from federated_intelligence.data.validators import check_group_disjoint, check_partition


@pytest.fixture
def labels():
    return np.random.default_rng(0).integers(0, 2, size=6000)


@pytest.mark.parametrize(
    "partitioner",
    [
        IIDPartitioner(20, seed=1),
        DirichletPartitioner(20, alpha=0.5, seed=1),
        DirichletPartitioner(20, alpha=0.1, seed=1),
        QuantitySkewPartitioner(20, sigma=1.0, seed=1),
    ],
)
def test_partition_covers_every_sample_exactly_once(partitioner, labels):
    part = partitioner.partition(labels)
    rep = check_partition(part, len(labels))
    assert rep.passed, rep.errors
    assert sum(len(v) for v in part.values()) == len(labels)


def test_partitioners_are_deterministic(labels):
    a = DirichletPartitioner(10, 0.5, seed=3).partition(labels)
    b = DirichletPartitioner(10, 0.5, seed=3).partition(labels)
    assert partition_checksum(a) == partition_checksum(b)
    c = DirichletPartitioner(10, 0.5, seed=4).partition(labels)
    assert partition_checksum(a) != partition_checksum(c)


def _label_spread(part, labels):
    return float(np.std([labels[i].mean() for i in part.values()]))


def test_smaller_alpha_means_stronger_label_skew(labels):
    spreads = [
        _label_spread(DirichletPartitioner(20, a, seed=2).partition(labels), labels)
        for a in (10.0, 1.0, 0.5, 0.1)
    ]
    assert spreads == sorted(spreads)  # monotone increasing as alpha decreases
    assert spreads[-1] > 3 * spreads[0]


def test_quantity_skew_has_unequal_sizes_but_balanced_labels(labels):
    part = QuantitySkewPartitioner(20, sigma=1.0, seed=2).partition(labels)
    sizes = np.array([len(v) for v in part.values()])
    assert sizes.std() / sizes.mean() > 0.4
    assert sizes.min() >= 10


def test_iid_sizes_nearly_equal(labels):
    sizes = [len(v) for v in IIDPartitioner(7, seed=0).partition(labels).values()]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_validates_alpha():
    with pytest.raises(ValueError):
        DirichletPartitioner(5, alpha=0)


def test_natural_partition_and_group_split():
    groups = np.repeat(np.arange(10), 20)
    labels = np.zeros(200, dtype=int)
    part = NaturalPartitioner(groups).partition(labels)
    assert len(part) == 10 and all(len(v) == 20 for v in part.values())
    seen, unseen = split_groups(groups, 0.3, seed=1)
    assert len(unseen) == 3
    assert check_group_disjoint(seen, unseen).passed
    assert not check_group_disjoint(seen, np.concatenate([unseen, seen[:1]])).passed


def test_check_partition_detects_problems():
    assert not check_partition({"a": np.array([0, 1]), "b": np.array([1, 2])}, 3).passed  # overlap
    assert not check_partition({"a": np.array([0, 1])}, 3).passed  # missing sample
    assert not check_partition({"a": np.array([0, 1, 2]), "b": np.array([], dtype=int)}, 3).passed


def test_manifest_contents(labels):
    p = DirichletPartitioner(5, 0.5, seed=9)
    part = p.partition(labels)
    m = build_manifest("dir_0_5_seed_9", "synthetic_federated_tabular", p, part)
    assert m["partition_method"] == "dirichlet" and m["alpha"] == 0.5 and m["seed"] == 9
    assert m["checksum_sha256"] == partition_checksum(part)
    total = sum(c["num_samples"] for c in m["clients"].values())
    assert total == len(labels)
    for c in m["clients"].values():
        assert c["train_samples"] + c["validation_samples"] + c["test_samples"] == c["num_samples"]
        assert len(c["indices"]) == c["num_samples"]
