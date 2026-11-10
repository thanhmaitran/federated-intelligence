import numpy as np
import pytest

from federated_intelligence.communication import (
    NetworkSimulator,
    NoCompression,
    TopKCompressor,
    build_compressor,
)
from federated_intelligence.communication.quantization import dequantize_int8, quantize_int8
from federated_intelligence.utils.serialization import (
    deserialize_params,
    payload_size,
    serialize_params,
)


@pytest.fixture
def update(rng):
    return [
        rng.normal(size=(40, 25)).astype(np.float32),
        rng.normal(size=(25,)).astype(np.float32),
        rng.normal(size=(25, 2)).astype(np.float32),
    ]


def test_serialization_roundtrip_and_size(update):
    payload = serialize_params(update)
    back = deserialize_params(payload)
    for a, b in zip(update, back, strict=True):
        np.testing.assert_array_equal(a, b)
    assert payload_size(update) == len(payload) > sum(u.nbytes for u in update)


def test_float32_baseline_is_lossless(update):
    c = NoCompression()
    out = c.decompress(c.compress(update), update)
    for a, b in zip(update, out, strict=True):
        np.testing.assert_array_equal(a, b)


def test_float16_halves_payload_with_small_error(update):
    base = len(NoCompression().compress(update))
    c = build_compressor("float16")
    payload = c.compress(update)
    assert 0.45 < len(payload) / base < 0.6
    out = c.decompress(payload, update)
    for a, b in zip(update, out, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=1e-3)


def test_int8_quarter_payload_bounded_error(update):
    base = len(NoCompression().compress(update))
    c = build_compressor("int8")
    payload = c.compress(update)
    assert len(payload) / base < 0.32
    out = c.decompress(payload, update)
    for a, b in zip(update, out, strict=True):
        assert np.max(np.abs(a - b)) <= np.max(np.abs(a)) / 127 / 2 + 1e-6  # half a quantum


def test_int8_primitive_zero_tensor():
    q, s = quantize_int8(np.zeros(5, np.float32))
    np.testing.assert_array_equal(dequantize_int8(q, s), np.zeros(5))


def test_topk_payload_scales_with_ratio_and_keeps_largest(update):
    base = len(NoCompression().compress(update))
    sizes = []
    for ratio in (0.5, 0.1, 0.01):
        c = TopKCompressor(ratio)
        payload = c.compress(update)
        sizes.append(len(payload))
        out = np.concatenate([x.ravel() for x in c.decompress(payload, update)])
        flat = np.concatenate([x.ravel() for x in update])
        k = int(round(ratio * flat.size))
        assert np.count_nonzero(out) <= k
        kept = np.flatnonzero(out)
        thresh = np.sort(np.abs(flat))[-k]
        assert np.all(np.abs(flat[kept]) >= thresh - 1e-7)
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] < 0.1 * base


def test_error_feedback_recovers_untransmitted_mass(update):
    """With EF, summed decoded updates over many identical rounds approach the true sum."""
    rounds = 30
    true_sum = [u * rounds for u in update]
    for ef in (False, True):
        c = TopKCompressor(0.1, error_feedback=ef)
        state: dict = {}
        acc = [np.zeros_like(u) for u in update]
        for _ in range(rounds):
            dec = c.decompress(c.compress(update, state), update)
            acc = [a + d for a, d in zip(acc, dec, strict=True)]
        err = np.sqrt(sum(np.sum((a - t) ** 2) for a, t in zip(acc, true_sum, strict=True)))
        if ef:
            err_ef = err
        else:
            err_no = err
    assert err_ef < 0.35 * err_no


def test_error_feedback_ignored_for_lossless():
    assert not build_compressor("none", error_feedback=True).error_feedback


def test_unknown_compressor_and_bad_ratio():
    with pytest.raises(ValueError):
        build_compressor("zstd")
    with pytest.raises(ValueError):
        TopKCompressor(0.0)


def test_network_time_increases_with_bytes_and_is_deterministic():
    ids = [f"c{i}" for i in range(5)]
    net = NetworkSimulator(ids, seed=1, latency_ms=20, bandwidth_mbps=1.0)
    small = net.simulate("c0", 1, 1_000, 1_000, 0.1)
    big = net.simulate("c0", 1, 1_000_000, 1_000_000, 0.1)
    assert big.time_s > small.time_s
    assert net.simulate("c0", 1, 1_000, 1_000, 0.1).time_s == small.time_s


def test_network_failures_and_deadline():
    ids = [f"c{i}" for i in range(50)]
    net = NetworkSimulator(ids, seed=2, failure_prob=0.5)
    ok = sum(net.simulate(c, 1, 10, 10, 0.0).ok for c in ids)
    assert 10 < ok < 40
    slow = NetworkSimulator(ids, seed=2, bandwidth_mbps=0.1, deadline_s=0.5)
    out = slow.simulate("c0", 1, 5_000_000, 5_000_000, 0.0)
    assert not out.ok and out.reason == "deadline_exceeded"


def test_availability_rate():
    ids = [f"c{i}" for i in range(200)]
    net = NetworkSimulator(ids, seed=3, availability=0.3)
    frac = np.mean([net.is_available(c, 4) for c in ids])
    assert 0.2 < frac < 0.4
