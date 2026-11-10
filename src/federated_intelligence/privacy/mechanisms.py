"""Client-level central DP aggregation (DP-FedAvg, McMahan et al. 2018).

Threat model and guarantee
--------------------------
* Trusted server (central DP): the server sees individual (clipped) client updates but only the
  noised aggregate is released as the new global model. If the server is untrusted, this
  mechanism does **not** protect against it - local DP or secure aggregation would be needed.
* Privacy unit: one client (all of its data). Adjacency: add/remove one client.
* Sampling: each client is included in a round independently with probability ``q``
  (Poisson sampling) - this is what the accountant assumes.
* Mechanism per round: clip each participating update to L2 norm ``C``, sum, add
  N(0, (sigma*C)^2 I), divide by the *fixed* expectation ``q*K`` (data-independent).
* Weights are uniform (unweighted clipped sum) so the sensitivity of the sum is exactly ``C``.
* Accounting: RDP of the sampled Gaussian mechanism, composed over rounds, converted to
  (epsilon, delta). See ``privacy.accounting``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from federated_intelligence.privacy.accounting import RDPAccountant
from federated_intelligence.privacy.clipping import clip_update
from federated_intelligence.privacy.noise import gaussian_noise_like

Params = list[np.ndarray]


@dataclass
class DPRoundStats:
    num_participants: int
    num_clipped: int
    mean_update_norm: float
    noise_std: float
    epsilon: float


class GaussianDPAggregator:
    """Clip + sum + Gaussian noise, with built-in accounting."""

    def __init__(
        self,
        clip_norm: float,
        noise_multiplier: float,
        sample_rate: float,
        num_clients: int,
        delta: float = 1e-5,
    ):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.num_clients = num_clients
        self.delta = delta
        self.accountant = RDPAccountant(noise_multiplier, sample_rate)

    @property
    def expected_participants(self) -> float:
        return self.sample_rate * self.num_clients

    def aggregate(
        self, updates: Sequence[Params], template: Sequence[np.ndarray], rng: np.random.Generator
    ) -> tuple[Params, DPRoundStats]:
        """Return the noised, normalised average update. Works with zero participants."""
        total = [np.zeros_like(t, dtype=np.float32) for t in template]
        norms, n_clipped = [], 0
        for u in updates:
            clipped, norm, was_clipped = clip_update(u, self.clip_norm)
            norms.append(norm)
            n_clipped += int(was_clipped)
            for acc, c in zip(total, clipped, strict=True):
                acc += c
        std = self.noise_multiplier * self.clip_norm
        noise = gaussian_noise_like(template, std, rng)
        denom = np.float32(self.expected_participants)
        avg = [((s + n) / denom).astype(np.float32) for s, n in zip(total, noise, strict=True)]
        self.accountant.step()
        stats = DPRoundStats(
            num_participants=len(updates),
            num_clipped=n_clipped,
            mean_update_norm=float(np.mean(norms)) if norms else 0.0,
            noise_std=std / float(denom),
            epsilon=self.accountant.get_epsilon(self.delta),
        )
        return avg, stats

    def describe(self) -> dict:
        return {
            **self.accountant.describe(self.delta),
            "clip_norm": self.clip_norm,
            "privacy_unit": "client",
            "adjacency": "add/remove one client",
            "sampling": "poisson",
            "threat_model": "central DP, trusted server",
        }
