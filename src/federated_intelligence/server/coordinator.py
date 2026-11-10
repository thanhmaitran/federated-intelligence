"""Central coordinator: runs the federated training loop.

Round lifecycle (see docs/architecture)::

    SELECT -> BROADCAST -> LOCAL TRAINING -> COLLECT -> VALIDATE -> AGGREGATE
           -> UPDATE GLOBAL MODEL -> EVALUATE -> LOG

Clients that do not deliver a valid update (dropped, failed, timed out, rejected, network
failure) are simply *excluded*; their update is never replaced by zeros, and the aggregation
weights are re-normalised over the clients that did participate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from federated_intelligence.algorithms.fedprox import build_algorithm
from federated_intelligence.clients.client import ClientUpdate
from federated_intelligence.clients.client_registry import ClientInfo, ClientRegistry
from federated_intelligence.clients.local_trainer import LocalTrainingConfig
from federated_intelligence.communication.compression import Compressor, NoCompression
from federated_intelligence.communication.network_simulator import NetworkSimulator
from federated_intelligence.data.federated_dataset import FederatedDataset
from federated_intelligence.evaluation.metrics import evaluate_model
from federated_intelligence.experiments.configuration import ExperimentConfig
from federated_intelligence.models.factory import get_parameters, set_parameters
from federated_intelligence.privacy.accounting import noise_multiplier_for_epsilon
from federated_intelligence.privacy.mechanisms import GaussianDPAggregator
from federated_intelligence.server.aggregator import apply_update, validate_update
from federated_intelligence.server.round_manager import RoundManager, build_selector
from federated_intelligence.utils.logging import get_logger
from federated_intelligence.utils.seeds import derive_seed
from federated_intelligence.utils.serialization import payload_size

logger = get_logger("federated_intelligence.coordinator")

Params = list[np.ndarray]
RoundCallback = Callable[[dict[str, Any]], None]


class Coordinator:
    def __init__(
        self,
        cfg: ExperimentConfig,
        dataset: FederatedDataset,
        model_fn: Callable,
        executor,
        compressor: Compressor,
        callbacks: list[RoundCallback] | None = None,
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.model_fn = model_fn
        self.executor = executor
        self.compressor = compressor
        self.callbacks = callbacks or []
        seed = cfg.experiment.seed

        self.registry = ClientRegistry()
        for cid in dataset.client_ids:
            self.registry.register(ClientInfo(cid, dataset.clients[cid].num_train))

        self.algorithm = build_algorithm(cfg.federated.algorithm, cfg.federated.mu)
        self.local_cfg = LocalTrainingConfig(
            local_epochs=cfg.training.local_epochs,
            batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
            mu=self.algorithm.mu,
        )

        self.network: NetworkSimulator | None = None
        if cfg.network.enabled:
            n = cfg.network
            self.network = NetworkSimulator(
                self.registry.ids,
                seed,
                latency_ms=n.latency_ms,
                bandwidth_mbps=n.bandwidth_mbps,
                heterogeneity=n.heterogeneity,
                failure_prob=n.failure_prob,
                availability=n.availability,
                deadline_s=n.deadline_s,
            )

        selection = cfg.federated.selection
        self.dp: GaussianDPAggregator | None = None
        if cfg.privacy.enabled:
            if selection != "poisson":
                logger.info("DP enabled: switching client selection to Poisson sampling")
                selection = "poisson"
            sigma = cfg.privacy.noise_multiplier
            if cfg.privacy.target_epsilon is not None:
                sigma = noise_multiplier_for_epsilon(
                    cfg.privacy.target_epsilon,
                    cfg.federated.participation_rate,
                    cfg.federated.num_rounds,
                    cfg.privacy.delta,
                )
            self.dp = GaussianDPAggregator(
                clip_norm=cfg.privacy.clip_norm,
                noise_multiplier=float(sigma),
                sample_rate=cfg.federated.participation_rate,
                num_clients=len(self.registry),
                delta=cfg.privacy.delta,
            )

        selector = build_selector(
            selection,
            self.registry.ids,
            cfg.federated.participation_rate,
            seed,
            is_available=self.network.is_available if self.network else None,
        )
        self.round_manager = RoundManager(selector, cfg.federated.dropout_rate, seed)

        self.eval_model = model_fn()
        self.global_params: Params = get_parameters(self.eval_model)
        # Reference: the same update serialized as one float32 vector (what 'none' sends).
        self.baseline_uplink_bytes = len(NoCompression().compress(self.global_params))
        self.val_x, self.val_y = dataset.pooled_val()
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ evaluation
    def evaluate_global(self, per_client: bool | None = None) -> dict[str, Any]:
        set_parameters(self.eval_model, self.global_params)
        test = evaluate_model(self.eval_model, self.dataset.global_test_x, self.dataset.global_test_y)
        val = evaluate_model(self.eval_model, self.val_x, self.val_y)
        row: dict[str, Any] = {f"test_{k}": v for k, v in test.items()}
        row.update({f"val_{k}": v for k, v in val.items()})
        if self.cfg.evaluation.per_client if per_client is None else per_client:
            per = self.executor.evaluate_clients(self.global_params, "test")
            acc = np.array([m["accuracy"] for m in per.values()])
            row.update(
                client_acc_mean=float(acc.mean()),
                client_acc_std=float(acc.std()),
                client_acc_min=float(acc.min()),
                client_acc_p10=float(np.percentile(acc, 10)),
            )
        return row

    def per_client_final(self) -> dict[str, dict[str, float]]:
        return self.executor.evaluate_clients(self.global_params, "test")

    # ------------------------------------------------------------------ communication
    def _downlink(self) -> tuple[Params, int]:
        """Parameters as clients receive them and the per-client downlink byte count."""
        if self.cfg.communication.downlink == "float16":
            sent = [p.astype(np.float16) for p in self.global_params]
            return [s.astype(np.float32) for s in sent], payload_size(sent)
        return self.global_params, payload_size(self.global_params)

    # ------------------------------------------------------------------ one round
    def run_round(self, r: int) -> dict[str, Any]:
        wall0 = time.perf_counter()
        plan = self.round_manager.plan(r)
        params_sent, down_per_client = self._downlink()
        downlink_bytes = down_per_client * len(plan.selected)

        updates = (
            self.executor.run_round(plan.participants, params_sent, self.local_cfg, r)
            if plan.participants
            else []
        )

        status_counts: dict[str, int] = {"dropped": len(plan.dropped)}
        valid: list[ClientUpdate] = []
        deltas: list[Params] = []
        sim_times: list[float] = []
        uplink_bytes = 0
        sent = 0
        for u in updates:
            if u.status != "ok":
                status_counts[u.status] = status_counts.get(u.status, 0) + 1
                continue
            sent += 1
            uplink_bytes += u.uplink_bytes  # the bytes were sent even if the round later rejects them
            if self.network is not None:
                out = self.network.simulate(u.client_id, r, down_per_client, u.uplink_bytes, u.compute_seconds)
                sim_times.append(out.time_s)
                if not out.ok:
                    status_counts[out.reason] = status_counts.get(out.reason, 0) + 1
                    continue
            try:
                delta = self.compressor.decompress(u.payload, self.global_params)
            except Exception:
                status_counts["rejected"] = status_counts.get("rejected", 0) + 1
                continue
            check = validate_update(delta, self.global_params)
            if not check.valid:
                logger.warning("round %d: rejected update from %s (%s)", r, u.client_id, check.reason)
                status_counts["rejected"] = status_counts.get("rejected", 0) + 1
                continue
            valid.append(u)
            deltas.append(delta)

        skipped = False
        dp_stats = None
        if self.dp is not None:
            rng = np.random.default_rng(derive_seed(self.cfg.experiment.seed, "dp_noise", r))
            avg, dp_stats = self.dp.aggregate(deltas, self.global_params, rng)
            self.global_params = apply_update(self.global_params, avg, self.cfg.federated.server_lr)
        elif valid:
            self.global_params = self.algorithm.aggregate(
                self.global_params, deltas, [u.num_samples for u in valid], self.cfg.federated.server_lr
            )
        else:
            skipped = True  # nobody delivered: the global model is left untouched

        train_loss = (
            float(np.average([u.train_loss for u in valid], weights=[u.num_samples for u in valid]))
            if valid
            else float("nan")
        )
        row: dict[str, Any] = {
            "round": r,
            "selected": len(plan.selected),
            "participants": len(valid),
            "sent_updates": sent,
            "failed_or_excluded": sum(v for k, v in status_counts.items()),
            "status_counts": status_counts,
            "skipped": skipped,
            "train_loss": train_loss,
            "uplink_bytes": uplink_bytes,
            "downlink_bytes": downlink_bytes,
            "compute_seconds": float(sum(u.compute_seconds for u in updates if u.status == "ok")),
            "compression_seconds": float(sum(u.compression_seconds for u in updates if u.status == "ok")),
            "sim_round_seconds": float(max(sim_times)) if sim_times else 0.0,
        }
        if dp_stats is not None:
            row.update(
                epsilon=dp_stats.epsilon,
                dp_clipped_fraction=dp_stats.num_clipped / max(dp_stats.num_participants, 1),
                dp_mean_update_norm=dp_stats.mean_update_norm,
            )
        if r % self.cfg.federated.eval_every == 0 or r == self.cfg.federated.num_rounds:
            ev = self.cfg.evaluation
            row.update(
                self.evaluate_global(
                    per_client=ev.per_client and (r % ev.per_client_every == 0 or r == self.cfg.federated.num_rounds)
                )
            )
        row["wall_seconds"] = time.perf_counter() - wall0
        return row

    # ------------------------------------------------------------------ full run
    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        init = {"round": 0, **self.evaluate_global()}
        self.history.append(init)
        for cb in self.callbacks:
            cb(init)
        for r in range(1, self.cfg.federated.num_rounds + 1):
            row = self.run_round(r)
            self.history.append(row)
            for cb in self.callbacks:
                cb(row)
        wall = time.perf_counter() - t0
        return self.summarize(wall)

    def summarize(self, wall_seconds: float) -> dict[str, Any]:
        rounds = [h for h in self.history if h["round"] > 0]
        evald = [h for h in self.history if "test_accuracy" in h]
        final = evald[-1]
        target = self.cfg.evaluation.target_accuracy
        rounds_to_target = next((h["round"] for h in evald if h.get("val_accuracy", 0) >= target), None)
        best_val = max(evald, key=lambda h: h["val_accuracy"])
        up = sum(h["uplink_bytes"] for h in rounds)
        down = sum(h["downlink_bytes"] for h in rounds)
        n_updates = sum(h["participants"] for h in rounds)
        n_sent = sum(h.get("sent_updates", h["participants"]) for h in rounds)
        summary: dict[str, Any] = {
            "final_round": final["round"],
            "final": {k: v for k, v in final.items() if k.startswith(("test_", "val_", "client_"))},
            "best_val_round": best_val["round"],
            "test_accuracy_at_best_val_round": best_val["test_accuracy"],
            "rounds_to_target_val_accuracy": rounds_to_target,
            "target_val_accuracy": target,
            "total_uplink_bytes": up,
            "total_downlink_bytes": down,
            "total_bytes": up + down,
            "mean_uplink_bytes_per_update": up / max(n_sent, 1),
            "baseline_float32_update_bytes": self.baseline_uplink_bytes,
            "uplink_compression_ratio": (self.baseline_uplink_bytes * max(n_sent, 1)) / max(up, 1),
            "total_participating_updates": n_updates,
            "skipped_rounds": sum(1 for h in rounds if h["skipped"]),
            "simulated_seconds_total": sum(h["sim_round_seconds"] for h in rounds),
            "compute_seconds_total": sum(h["compute_seconds"] for h in rounds),
            "wall_seconds": wall_seconds,
            "executor": self.executor.name,
        }
        if self.dp is not None:
            summary["privacy"] = self.dp.describe()
        return summary
