"""Synthetic federated tabular data with controllable heterogeneity.

Data-generating process for client k (see docs/methodology/synthetic-data.md)::

    X ~ N(mu_k, diag(scale_k^2))
    P(Y=1 | x, k) = sigmoid(w_k^T x + b_k),   w_k = w_0 + eps_k
    b_k calibrated so that E[P(Y=1|X,k)] = p_k,   p_k ~ Beta(a, a) under label skew

The global test set is drawn from the *population mixture* (equal-weight mixture of all
client distributions) with an independent RNG stream. It is therefore never built from
client-local test splits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from federated_intelligence.data.federated_dataset import (
    ClientSplit,
    FederatedDataset,
    split_indices,
)
from federated_intelligence.data.generators.base import BaseGenerator
from federated_intelligence.data.generators.client_parameters import ClientParameters
from federated_intelligence.data.generators.feature_distribution import (
    sample_feature_params,
    sample_features,
)
from federated_intelligence.data.generators.generator_config import (
    DATASET_NAME,
    GENERATOR_VERSION,
    GeneratorConfig,
)
from federated_intelligence.data.generators.label_distribution import (
    calibrate_bias,
    sample_positive_rates,
    sample_sizes,
)
from federated_intelligence.utils.seeds import derive_seed


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TabularGenerator(BaseGenerator):
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.client_params: list[ClientParameters] = []

    # ------------------------------------------------------------------ parameters
    def build_client_parameters(self) -> list[ClientParameters]:
        cfg = self.config
        rng = np.random.default_rng(derive_seed(cfg.seed, "params"))
        w0 = rng.normal(0.0, cfg.weight_scale, size=cfg.num_features)
        # Unit-norm direction scaled so the logit has std ~ 2*weight_scale for N(0, I) features.
        w0 = w0 / np.linalg.norm(w0) * 2.0 * cfg.weight_scale
        rates = sample_positive_rates(cfg, rng)
        sizes = sample_sizes(cfg, rng)
        means, scales = sample_feature_params(cfg, rng)

        params: list[ClientParameters] = []
        for k in range(cfg.num_clients):
            w_k = w0 + rng.normal(0.0, cfg.concept_noise, size=cfg.num_features)
            probe_rng = np.random.default_rng(derive_seed(cfg.seed, "probe", k))
            probe = sample_features(means[k], scales[k], 4000, probe_rng)
            b_k = calibrate_bias(probe, w_k, float(rates[k]))
            params.append(
                ClientParameters(
                    client_id=f"client_{k:03d}",
                    num_samples=int(sizes[k]),
                    positive_rate=float(rates[k]),
                    mean=means[k],
                    scale=scales[k],
                    weights=w_k,
                    bias=float(b_k),
                )
            )
        self.client_params = params
        return params

    # ------------------------------------------------------------------ sampling
    def _sample(
        self, p: ClientParameters, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x = sample_features(p.mean, p.scale, n, rng)
        prob = _sigmoid(x @ p.weights + p.bias)
        y = (rng.random(n) < prob).astype(np.int64)
        if self.config.label_noise > 0:
            flip = rng.random(n) < self.config.label_noise
            y = np.where(flip, 1 - y, y)
        return x, y

    def generate(self) -> FederatedDataset:
        cfg = self.config
        params = self.build_client_parameters()
        clients: dict[str, ClientSplit] = {}
        fractions = (cfg.train_fraction, cfg.val_fraction, cfg.test_fraction)

        for k, p in enumerate(params):
            rng = np.random.default_rng(derive_seed(cfg.seed, "client", k))
            x, y = self._sample(p, p.num_samples, rng)
            tr, va, te = split_indices(len(y), fractions, rng)
            clients[p.client_id] = ClientSplit(
                p.client_id, x[tr], y[tr], x[va], y[va], x[te], y[te]
            )

        # Independent global test set from the population mixture.
        grng = np.random.default_rng(derive_seed(cfg.seed, "global_test"))
        per_client = max(1, cfg.global_test_size // cfg.num_clients)
        gx, gy = [], []
        for p in params:
            x, y = self._sample(p, per_client, grng)
            gx.append(x)
            gy.append(y)

        return FederatedDataset(
            name=DATASET_NAME,
            clients=clients,
            global_test_x=np.concatenate(gx),
            global_test_y=np.concatenate(gy),
            num_features=cfg.num_features,
            num_classes=cfg.num_classes,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict:
        cfg = self.config
        return {
            "dataset": DATASET_NAME,
            "seed": cfg.seed,
            "num_clients": cfg.num_clients,
            "num_features": cfg.num_features,
            "num_classes": cfg.num_classes,
            "scenario": cfg.scenario.value,
            "generator_version": GENERATOR_VERSION,
            "config": json.loads(cfg.model_dump_json()),
        }

    # ------------------------------------------------------------------ persistence
    def write(self, out_dir: Path, dataset: FederatedDataset | None = None) -> dict:
        """Write Parquet files + manifest (with SHA-256 checksums) to ``out_dir``."""
        dataset = dataset or self.generate()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = [f"f{i:02d}" for i in range(self.config.num_features)]

        frames = []
        for cid in dataset.client_ids:
            c = dataset.clients[cid]
            for split in ("train", "val", "test"):
                x, y = getattr(c, f"x_{split}"), getattr(c, f"y_{split}")
                df = pd.DataFrame(x, columns=cols)
                df["label"] = y
                df["split"] = split
                df["client_id"] = cid
                frames.append(df)
        clients_path = out_dir / "clients.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(clients_path, index=False)

        gdf = pd.DataFrame(dataset.global_test_x, columns=cols)
        gdf["label"] = dataset.global_test_y
        global_path = out_dir / "global_test.parquet"
        gdf.to_parquet(global_path, index=False)

        manifest = dict(dataset.metadata)
        manifest["files"] = {
            p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in (clients_path, global_path)
        }
        manifest["client_parameters"] = [p.to_dict() for p in self.client_params]
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset(config: GeneratorConfig) -> FederatedDataset:
    return TabularGenerator(config).generate()
