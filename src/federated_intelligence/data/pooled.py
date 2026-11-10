"""Build federated datasets by *partitioning a pooled population* (Dirichlet / IID / quantity).

This complements the native per-client generator: here one population distribution is
sampled once and the partitioner decides which client owns which sample. It is the setup
used for the Dirichlet-alpha sensitivity study (EXP-02/03).
"""

from __future__ import annotations

import numpy as np

from federated_intelligence.data.federated_dataset import (
    ClientSplit,
    FederatedDataset,
    split_indices,
)
from federated_intelligence.data.generators.generator_config import GeneratorConfig, Scenario
from federated_intelligence.data.generators.tabular_generator import TabularGenerator
from federated_intelligence.data.partitioners.base import BasePartitioner
from federated_intelligence.data.partitioners.dirichlet import DirichletPartitioner
from federated_intelligence.data.partitioners.iid import IIDPartitioner
from federated_intelligence.data.partitioners.quantity import QuantitySkewPartitioner
from federated_intelligence.utils.seeds import derive_seed


def sample_population(
    cfg: GeneratorConfig, n: int, stream: str = "population"
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` points from a single IID population (all clients share P).

    ``stream`` selects an independent RNG stream; the population parameters are identical.
    """
    pop_cfg = cfg.model_copy(update={"scenario": Scenario.IID, "num_clients": 2})
    gen = TabularGenerator(pop_cfg)
    params = gen.build_client_parameters()[0]
    rng = np.random.default_rng(derive_seed(cfg.seed, stream))
    return gen._sample(params, n, rng)


def make_partitioner(method: str, num_clients: int, seed: int, **kw) -> BasePartitioner:
    if method == "iid":
        return IIDPartitioner(num_clients, seed)
    if method == "dirichlet":
        return DirichletPartitioner(num_clients, alpha=kw["alpha"], seed=seed)
    if method == "quantity_skew":
        return QuantitySkewPartitioner(num_clients, sigma=kw.get("sigma", 1.0), seed=seed)
    raise ValueError(f"Unknown partition method: {method}")


def build_partitioned_dataset(
    cfg: GeneratorConfig,
    method: str,
    total_samples: int,
    **partition_kwargs,
) -> FederatedDataset:
    x, y = sample_population(cfg, total_samples)
    partitioner = make_partitioner(method, cfg.num_clients, cfg.seed, **partition_kwargs)
    partition = partitioner.partition(y)

    fractions = (cfg.train_fraction, cfg.val_fraction, cfg.test_fraction)
    clients: dict[str, ClientSplit] = {}
    for cid, idx in partition.items():
        rng = np.random.default_rng(derive_seed(cfg.seed, "split", cid))
        tr, va, te = split_indices(len(idx), fractions, rng)
        clients[cid] = ClientSplit(
            cid, x[idx[tr]], y[idx[tr]], x[idx[va]], y[idx[va]], x[idx[te]], y[idx[te]]
        )

    gx, gy = sample_population(cfg, cfg.global_test_size, stream="population_global_test")
    meta = {
        "dataset": "synthetic_federated_tabular",
        "partition_method": partitioner.method,
        **partitioner.params(),
        "seed": cfg.seed,
        "num_clients": cfg.num_clients,
        "total_samples": total_samples,
    }
    return FederatedDataset(
        f"synthetic_{method}", clients, gx, gy, cfg.num_features, cfg.num_classes, meta
    )
