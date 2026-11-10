"""Renyi-DP accountant for the Sampled Gaussian Mechanism (SGM).

Implements the integer-order RDP bound of Mironov, Talwar & Zhang (2019), "Renyi Differential
Privacy of the Sampled Gaussian Mechanism":

    A_a = sum_{k=0}^{a} C(a, k) (1-q)^{a-k} q^k exp((k^2 - k) / (2 sigma^2))
    RDP(a) = log(A_a) / (a - 1)                       (per step, integer a >= 2)

RDP composes additively over steps and is converted to (epsilon, delta)-DP with the improved
conversion of Balle et al. (2020):

    eps = RDP_total(a) + log((a-1)/a) - (log(delta) + log(a)) / (a - 1),   minimised over a.

Privacy unit in this project: one *client* (add/remove adjacency on the set of clients), with
Poisson client sampling at rate q each round. The guarantee only holds when the training loop
really implements that mechanism (see ``privacy.mechanisms``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

DEFAULT_ORDERS: tuple[int, ...] = tuple(range(2, 65)) + (80, 96, 128, 192, 256, 384, 512)


def _log_binom(n: int, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of one SGM step at integer order ``alpha`` (>= 2)."""
    if alpha < 2 or int(alpha) != alpha:
        raise ValueError("alpha must be an integer >= 2")
    if sigma <= 0:
        return math.inf
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * sigma**2)
    k = np.arange(alpha + 1, dtype=np.float64)
    log_terms = (
        _log_binom(alpha, k)
        + (alpha - k) * math.log1p(-q)
        + k * math.log(q)
        + (k * k - k) / (2.0 * sigma**2)
    )
    return float(logsumexp(log_terms)) / (alpha - 1)


def rdp_to_eps(rdp: np.ndarray, orders: np.ndarray, delta: float) -> tuple[float, int]:
    """Convert accumulated RDP values to (epsilon, optimal order)."""
    orders = np.asarray(orders, dtype=np.float64)
    eps = rdp + np.log((orders - 1) / orders) - (math.log(delta) + np.log(orders)) / (orders - 1)
    eps = np.where(np.isfinite(eps), eps, np.inf)
    i = int(np.nanargmin(eps))
    return float(max(eps[i], 0.0)), int(orders[i])


class RDPAccountant:
    """Tracks cumulative privacy loss of repeated SGM steps with fixed (q, sigma)."""

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        orders: tuple[int, ...] = DEFAULT_ORDERS,
    ):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.orders = np.asarray(orders)
        self._per_step = np.array(
            [rdp_sampled_gaussian(sample_rate, noise_multiplier, int(a)) for a in orders]
        )
        self.steps = 0

    def step(self, n: int = 1) -> None:
        self.steps += n

    def get_epsilon(self, delta: float) -> float:
        if self.steps == 0:
            return 0.0
        return rdp_to_eps(self._per_step * self.steps, self.orders, delta)[0]

    def get_epsilon_and_order(self, delta: float) -> tuple[float, int]:
        return rdp_to_eps(self._per_step * self.steps, self.orders, delta)

    def describe(self, delta: float) -> dict:
        eps, order = self.get_epsilon_and_order(delta)
        return {
            "accountant": "rdp_sampled_gaussian",
            "epsilon": eps,
            "delta": delta,
            "best_order": order,
            "steps": self.steps,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
        }


def epsilon_after(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float
) -> float:
    acc = RDPAccountant(noise_multiplier, sample_rate)
    acc.step(steps)
    return acc.get_epsilon(delta)


def noise_multiplier_for_epsilon(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
    tol: float = 1e-3,
) -> float:
    """Smallest sigma (up to ``tol``) whose accounted epsilon after ``steps`` is <= target."""
    lo, hi = 0.05, 1.0
    while epsilon_after(hi, sample_rate, steps, delta) > target_epsilon:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError("target epsilon unreachable")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if epsilon_after(mid, sample_rate, steps, delta) > target_epsilon:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi
