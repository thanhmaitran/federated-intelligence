import math

import numpy as np
import pytest

from federated_intelligence.privacy import (
    GaussianDPAggregator,
    RDPAccountant,
    clip_update,
    epsilon_after,
    gaussian_noise_like,
    global_l2_norm,
    noise_multiplier_for_epsilon,
    rdp_sampled_gaussian,
)
from federated_intelligence.privacy.accounting import DEFAULT_ORDERS


def test_clipping_bounds_norm_and_preserves_direction(rng):
    u = [rng.normal(size=(5, 4)).astype(np.float32) * 10, rng.normal(size=(4,)).astype(np.float32)]
    clipped, norm, was = clip_update(u, 1.0)
    assert was and norm > 1.0
    assert math.isclose(global_l2_norm(clipped), 1.0, rel_tol=1e-4)
    flat_u = np.concatenate([x.ravel() for x in u])
    flat_c = np.concatenate([x.ravel() for x in clipped])
    cos = flat_u @ flat_c / (np.linalg.norm(flat_u) * np.linalg.norm(flat_c))
    assert cos > 0.9999


def test_small_updates_not_modified(rng):
    u = [rng.normal(size=(3,)).astype(np.float32) * 0.01]
    clipped, _, was = clip_update(u, 1.0)
    assert not was
    np.testing.assert_array_equal(clipped[0], u[0])


def test_zero_update_and_invalid_clip(rng):
    clipped, norm, was = clip_update([np.zeros(3, np.float32)], 1.0)
    assert norm == 0 and not was
    with pytest.raises(ValueError):
        clip_update([np.ones(3, np.float32)], 0.0)


def test_noise_std_matches(rng):
    noise = gaussian_noise_like([np.zeros(200_000, np.float32)], 2.5, rng)[0]
    assert abs(noise.std() - 2.5) < 0.02
    assert abs(noise.mean()) < 0.02


def test_rdp_q1_reduces_to_gaussian_mechanism():
    for sigma in (0.8, 1.5, 4.0):
        for a in (2, 5, 20):
            assert math.isclose(rdp_sampled_gaussian(1.0, sigma, a), a / (2 * sigma**2), rel_tol=1e-9)


def test_rdp_q0_is_free():
    assert rdp_sampled_gaussian(0.0, 1.0, 8) == 0.0


def test_subsampling_amplifies_privacy():
    assert rdp_sampled_gaussian(0.1, 1.0, 8) < rdp_sampled_gaussian(1.0, 1.0, 8)


def test_epsilon_monotonicity():
    d = 1e-5
    assert epsilon_after(1.0, 0.1, 50, d) < epsilon_after(1.0, 0.1, 100, d)  # more rounds
    assert epsilon_after(2.0, 0.1, 50, d) < epsilon_after(1.0, 0.1, 50, d)  # more noise
    assert epsilon_after(1.0, 0.05, 50, d) < epsilon_after(1.0, 0.2, 50, d)  # smaller q
    assert epsilon_after(1.0, 0.1, 50, 1e-3) < epsilon_after(1.0, 0.1, 50, 1e-7)  # larger delta


def test_zero_steps_zero_epsilon():
    assert RDPAccountant(1.0, 0.1).get_epsilon(1e-5) == 0.0


def test_noise_multiplier_search_hits_target():
    for target in (1.0, 4.0, 8.0):
        sigma = noise_multiplier_for_epsilon(target, 0.2, 40, 1e-5)
        eps = epsilon_after(sigma, 0.2, 40, 1e-5)
        assert eps <= target + 1e-9
        assert eps > 0.9 * target  # tight, not wildly conservative


def test_accountant_matches_opacus_reference():
    """Independent check against Opacus' RDP analysis (skipped when Opacus is unavailable)."""
    orp = pytest.importorskip("opacus.accountants.analysis.rdp")
    orders = list(DEFAULT_ORDERS)
    for q, sigma, steps in [(0.01, 1.1, 500), (0.1, 2.0, 100), (0.5, 3.0, 30), (1.0, 5.0, 10)]:
        acc = RDPAccountant(sigma, q)
        acc.step(steps)
        ref_rdp = orp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)
        ref_eps, _ = orp.get_privacy_spent(orders=orders, rdp=ref_rdp, delta=1e-5)
        assert math.isclose(acc.get_epsilon(1e-5), ref_eps, rel_tol=1e-6)


def test_dp_aggregator_zero_noise_is_clipped_mean(rng):
    template = [np.zeros((3,), np.float32)]
    agg = GaussianDPAggregator(clip_norm=1.0, noise_multiplier=0.0, sample_rate=0.5, num_clients=4)
    updates = [[np.array([3.0, 0, 0], np.float32)], [np.array([0, 0.5, 0], np.float32)]]
    out, stats = agg.aggregate(updates, template, rng)
    # clipped: [1,0,0] and [0,0.5,0]; divisor = q*K = 2
    np.testing.assert_allclose(out[0], [0.5, 0.25, 0.0], atol=1e-6)
    assert stats.num_clipped == 1 and stats.num_participants == 2


def test_dp_aggregator_noise_scale_and_empty_round():
    template = [np.zeros((50_000,), np.float32)]
    agg = GaussianDPAggregator(clip_norm=2.0, noise_multiplier=1.5, sample_rate=0.25, num_clients=40)
    out, stats = agg.aggregate([], template, np.random.default_rng(0))  # zero participants
    expected_std = 1.5 * 2.0 / (0.25 * 40)
    assert abs(out[0].std() - expected_std) < 0.01 * expected_std + 1e-3
    assert math.isclose(stats.noise_std, expected_std)
    assert agg.accountant.steps == 1  # empty rounds still cost privacy budget


def test_sensitivity_of_sum_is_clip_norm(rng):
    """Adding/removing one client changes the clipped sum by at most C in L2."""
    template = [np.zeros((20,), np.float32)]
    agg = GaussianDPAggregator(1.0, 0.0, 1.0, 5)
    ups = [[rng.normal(size=20).astype(np.float32) * 5] for _ in range(5)]
    with_all, _ = agg.aggregate(ups, template, rng)
    without, _ = agg.aggregate(ups[:-1], template, rng)
    diff = np.linalg.norm((with_all[0] - without[0]) * 5)  # undo the /(qK) divisor
    assert diff <= 1.0 + 1e-5


def test_describe_reports_assumptions():
    agg = GaussianDPAggregator(1.0, 1.0, 0.1, 50)
    agg.accountant.step(10)
    d = agg.describe()
    assert d["privacy_unit"] == "client" and d["sampling"] == "poisson"
    assert d["delta"] == 1e-5 and d["epsilon"] > 0 and "trusted server" in d["threat_model"]
