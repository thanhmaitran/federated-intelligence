"""Network simulator.

Produces *simulated* communication/round time. It is deliberately separate from the wall-clock
time of the experiment: simulated time models what the round would cost on the configured
network, wall-clock time is how long the simulation really took.

Per-client link: latency_k (ms) and bandwidth_k (Mbit/s) are drawn once, log-normally around the
configured means. Round time of a synchronous round is the max over participating clients, capped
by an optional deadline (clients that would exceed it are dropped as stragglers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from federated_intelligence.utils.seeds import derive_seed


@dataclass
class LinkOutcome:
    ok: bool
    reason: str
    time_s: float


class NetworkSimulator:
    def __init__(
        self,
        client_ids: list[str],
        seed: int,
        latency_ms: float = 50.0,
        bandwidth_mbps: float = 10.0,
        heterogeneity: float = 0.5,
        failure_prob: float = 0.0,
        availability: float = 1.0,
        deadline_s: float | None = None,
        compute_scale: float = 1.0,
    ):
        rng = np.random.default_rng(derive_seed(seed, "network"))
        n = len(client_ids)
        jitter = lambda: rng.lognormal(0.0, heterogeneity, size=n)  # noqa: E731
        self.latency_s = dict(zip(client_ids, latency_ms / 1000.0 * jitter(), strict=True))
        self.bandwidth_bps = dict(zip(client_ids, bandwidth_mbps * 1e6 * jitter(), strict=True))
        self.speed = dict(zip(client_ids, 1.0 / jitter(), strict=True))
        self.failure_prob = failure_prob
        self.availability = availability
        self.deadline_s = deadline_s
        self.compute_scale = compute_scale
        self.seed = seed

    def is_available(self, client_id: str, round_idx: int) -> bool:
        rng = np.random.default_rng(derive_seed(self.seed, "avail", round_idx, client_id))
        return bool(rng.random() < self.availability)

    def simulate(
        self,
        client_id: str,
        round_idx: int,
        downlink_bytes: int,
        uplink_bytes: int,
        compute_s: float,
    ) -> LinkOutcome:
        rng = np.random.default_rng(derive_seed(self.seed, "link", round_idx, client_id))
        lat, bw = self.latency_s[client_id], self.bandwidth_bps[client_id]
        t_down = lat + downlink_bytes * 8.0 / bw
        t_up = lat + uplink_bytes * 8.0 / bw
        t_compute = compute_s * self.compute_scale / self.speed[client_id]
        total = t_down + t_compute + t_up
        if rng.random() < self.failure_prob:
            return LinkOutcome(False, "communication_failure", total)
        if self.deadline_s is not None and total > self.deadline_s:
            return LinkOutcome(False, "deadline_exceeded", self.deadline_s)
        return LinkOutcome(True, "ok", total)
