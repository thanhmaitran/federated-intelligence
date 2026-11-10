import numpy as np
import pytest
import torch

from federated_intelligence.algorithms import FedAvg, FedProx, build_algorithm
from federated_intelligence.clients.local_trainer import LocalTrainingConfig, train_local
from federated_intelligence.models import get_parameters, model_factory, set_parameters
from federated_intelligence.server import (
    AggregationError,
    apply_update,
    fedavg_params,
    normalized_weights,
    validate_update,
    weighted_average,
)


def rand_params(rng, scale=1.0):
    return [rng.normal(size=(4, 3)).astype(np.float32) * scale, rng.normal(size=(3,)).astype(np.float32)]


def test_single_client_fedavg_equals_that_client(rng):
    p = rand_params(rng)
    out = fedavg_params([p], [123])
    for a, b in zip(out, p, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_identical_clients_keep_parameters(rng):
    p = rand_params(rng)
    out = fedavg_params([p, [x.copy() for x in p], [x.copy() for x in p]], [10, 50, 3])
    for a, b in zip(out, p, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_weights_proportional_to_samples(rng):
    a, b = rand_params(rng), rand_params(rng)
    out = fedavg_params([a, b], [1, 3])
    for o, x, y in zip(out, a, b, strict=True):
        np.testing.assert_allclose(o, 0.25 * x + 0.75 * y, rtol=1e-5, atol=1e-6)


def test_delta_form_equals_weight_form(rng):
    g = rand_params(rng)
    clients = [rand_params(rng) for _ in range(4)]
    n = [5, 9, 20, 2]
    direct = fedavg_params(clients, n)
    deltas = [[c - gg for c, gg in zip(cl, g, strict=True)] for cl in clients]
    via_delta = FedAvg().aggregate(g, deltas, n)
    for a, b in zip(direct, via_delta, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_empty_selection_raises():
    with pytest.raises(AggregationError):
        weighted_average([], np.array([]))
    with pytest.raises(AggregationError):
        normalized_weights([])
    with pytest.raises(AggregationError):
        normalized_weights([0, 0])
    with pytest.raises(AggregationError):
        normalized_weights([-1, 2])


def test_weights_must_sum_to_one(rng):
    with pytest.raises(AggregationError):
        weighted_average([rand_params(rng)], np.array([0.5]))


def test_validate_update_catches_bad_updates(rng):
    t = rand_params(rng)
    assert validate_update(t, t).valid
    assert not validate_update(t[:1], t).valid  # missing tensor
    assert not validate_update([t[0], np.zeros((5,), np.float32)], t).valid  # shape
    bad = [t[0].copy(), t[1].copy()]
    bad[0][0, 0] = np.nan
    assert not validate_update(bad, t).valid  # non-finite
    assert not validate_update([x.astype(np.int32) for x in t], t).valid  # dtype


def test_absent_client_is_excluded_not_zeroed(rng):
    """Aggregating over 2 present clients != treating a missing third client as a zero update."""
    d1, d2 = rand_params(rng), rand_params(rng)
    g = rand_params(rng)
    right = FedAvg().aggregate(g, [d1, d2], [10, 10])
    zero = [np.zeros_like(x) for x in d1]
    wrong = FedAvg().aggregate(g, [d1, d2, zero], [10, 10, 10])
    assert not np.allclose(right[0], wrong[0])


def test_apply_update_server_lr(rng):
    g, d = rand_params(rng), rand_params(rng)
    out = apply_update(g, d, server_lr=0.5)
    np.testing.assert_allclose(out[0], g[0] + 0.5 * d[0], rtol=1e-6)


def test_algorithm_registry():
    assert build_algorithm("fedavg").mu == 0
    assert isinstance(build_algorithm("fedprox", 0.1), FedProx)
    assert build_algorithm("fedprox", 0.1).mu == 0.1
    with pytest.raises(ValueError):
        build_algorithm("nope")
    with pytest.raises(ValueError):
        FedProx(-1.0)


def _local_run(mu, x, y, seed=0):
    fn = model_factory("mlp", num_features=x.shape[1], num_classes=2, hidden=[8], init_seed=1)
    model = fn()
    g = get_parameters(model)
    cfg = LocalTrainingConfig(local_epochs=2, batch_size=16, learning_rate=0.1, mu=mu)
    train_local(model, x, y, cfg, g, seed)
    return g, get_parameters(model)


def test_fedprox_mu0_matches_fedavg_local_training_exactly(rng):
    x = rng.normal(size=(64, 5)).astype(np.float32)
    y = rng.integers(0, 2, 64)
    _, a = _local_run(0.0, x, y)
    _, b = _local_run(0.0, x, y)
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(u, v)


def test_fedprox_mu_pulls_towards_global(rng):
    x = rng.normal(size=(64, 5)).astype(np.float32)
    y = rng.integers(0, 2, 64)
    g, w0 = _local_run(0.0, x, y)
    _, w_prox = _local_run(10.0, x, y)

    def dist(w):
        return float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(w, g, strict=True))))

    assert dist(w_prox) < dist(w0)


def test_proximal_gradient_is_mu_times_difference():
    """d/dw [(mu/2)||w - w_t||^2] = mu (w - w_t)."""
    from federated_intelligence.clients.local_trainer import proximal_term

    model = torch.nn.Linear(3, 2)
    anchor = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    mu = 0.7
    (mu * proximal_term(model, anchor)).backward()
    for _, p in model.named_parameters():
        torch.testing.assert_close(p.grad, mu * p.detach())


def test_parameter_roundtrip():
    fn = model_factory("mlp", num_features=6, num_classes=2, hidden=[4], init_seed=3)
    m = fn()
    params = get_parameters(m)
    m2 = fn()
    set_parameters(m2, [p + 1 for p in params])
    for a, b in zip(get_parameters(m2), params, strict=True):
        np.testing.assert_allclose(a, b + 1, rtol=1e-6)
    with pytest.raises(ValueError):
        set_parameters(m2, params[:-1])


def test_model_factory_gives_identical_init():
    fn = model_factory("mlp", num_features=6, num_classes=2, hidden=[4], init_seed=5)
    for a, b in zip(get_parameters(fn()), get_parameters(fn()), strict=True):
        np.testing.assert_array_equal(a, b)
