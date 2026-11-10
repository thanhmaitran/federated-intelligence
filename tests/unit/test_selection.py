import numpy as np
import pytest

from federated_intelligence.server import (
    AvailabilitySelector,
    PoissonSelector,
    RandomSelector,
    RoundManager,
    build_selector,
)

IDS = [f"client_{i:03d}" for i in range(40)]


@pytest.mark.parametrize("rate", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_random_selector_participation_rate(rate):
    sel = RandomSelector(IDS, rate, seed=1)
    for r in range(5):
        chosen = sel.select(r)
        assert len(chosen) == max(1, round(rate * len(IDS)))
        assert len(set(chosen)) == len(chosen)
        assert set(chosen) <= set(IDS)


def test_selection_is_deterministic_and_varies_across_rounds():
    sel = RandomSelector(IDS, 0.25, seed=1)
    assert sel.select(3) == sel.select(3)
    assert sel.select(3) != sel.select(4)
    assert RandomSelector(IDS, 0.25, seed=2).select(3) != sel.select(3)


def test_every_client_eventually_selected():
    sel = RandomSelector(IDS, 0.25, seed=0)
    seen = set()
    for r in range(60):
        seen |= set(sel.select(r))
    assert seen == set(IDS)


def test_poisson_expected_size_and_possible_empty():
    sel = PoissonSelector(IDS, 0.25, seed=5)
    sizes = [len(sel.select(r)) for r in range(400)]
    assert abs(np.mean(sizes) - 10) < 0.7
    tiny = PoissonSelector(IDS[:3], 0.05, seed=5)
    assert any(len(tiny.select(r)) == 0 for r in range(200))


def test_availability_selector_only_picks_available():
    available = {c for c in IDS if int(c[-3:]) % 2 == 0}
    sel = AvailabilitySelector(IDS, 0.5, 0, lambda c, r: c in available)
    for r in range(10):
        chosen = sel.select(r)
        assert set(chosen) <= available
        assert len(chosen) == min(len(available), 20)
    none = AvailabilitySelector(IDS, 0.5, 0, lambda c, r: False)
    assert none.select(0) == []


def test_dropout_rate_and_participants():
    mgr = RoundManager(RandomSelector(IDS, 1.0, 0), dropout_rate=0.3, seed=9)
    plans = [mgr.plan(r) for r in range(100)]
    frac = np.mean([len(p.dropped) / len(p.selected) for p in plans])
    assert 0.25 < frac < 0.35
    for p in plans[:5]:
        assert set(p.participants).isdisjoint(p.dropped)
        assert set(p.participants) | set(p.dropped) == set(p.selected)


def test_no_dropout_means_everyone_participates():
    plan = RoundManager(RandomSelector(IDS, 0.5, 0), 0.0, 0).plan(1)
    assert plan.dropped == [] and plan.participants == plan.selected


def test_invalid_rate_and_kind():
    with pytest.raises(ValueError):
        RandomSelector(IDS, 0.0, 0)
    with pytest.raises(ValueError):
        build_selector("nope", IDS, 0.5, 0)
