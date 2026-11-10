"""Statistical heterogeneity metrics between clients (Phase 5.2).

All metrics are computed on client *training* data.

* Label distribution: class proportions per client.
* Jensen-Shannon divergence (base 2, squared distance => value in [0, 1]).
* Feature mean distance: mean pairwise L2 distance between client feature means.
* Feature covariance distance: mean pairwise Frobenius distance between client covariances.
* Sample-count coefficient of variation CV = std(n_k) / mean(n_k).
* Client similarity matrix: 1 - sqrt(JSD) between label distributions.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial.distance import jensenshannon

from federated_intelligence.data.federated_dataset import FederatedDataset


def label_distribution_matrix(ds: FederatedDataset) -> np.ndarray:
    """(K, C) matrix of per-client class proportions."""
    rows = []
    for cid in ds.client_ids:
        y = ds.clients[cid].y_train
        counts = np.bincount(y, minlength=ds.num_classes).astype(float)
        rows.append(counts / max(counts.sum(), 1.0))
    return np.vstack(rows)


def jsd_matrix(dist: np.ndarray) -> np.ndarray:
    k = len(dist)
    m = np.zeros((k, k))
    for i, j in combinations(range(k), 2):
        d = jensenshannon(dist[i], dist[j], base=2) ** 2
        m[i, j] = m[j, i] = 0.0 if np.isnan(d) else d
    return m


def _mean_pairwise(values: list[np.ndarray]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.mean([np.linalg.norm(a - b) for a, b in combinations(values, 2)]))


def feature_mean_distance(ds: FederatedDataset) -> float:
    means = [ds.clients[c].x_train.mean(axis=0) for c in ds.client_ids]
    return _mean_pairwise(means)


def feature_covariance_distance(ds: FederatedDataset) -> float:
    covs = [np.cov(ds.clients[c].x_train, rowvar=False) for c in ds.client_ids]
    return _mean_pairwise(covs)


def sample_count_cv(ds: FederatedDataset) -> float:
    n = np.array([ds.clients[c].num_train for c in ds.client_ids], dtype=float)
    return float(n.std() / n.mean())


def similarity_matrix(dist: np.ndarray) -> np.ndarray:
    return 1.0 - np.sqrt(jsd_matrix(dist))


def heterogeneity_summary(ds: FederatedDataset) -> dict:
    dist = label_distribution_matrix(ds)
    jsd = jsd_matrix(dist)
    iu = np.triu_indices(len(dist), k=1)
    return {
        "num_clients": len(dist),
        "mean_pairwise_label_jsd": float(jsd[iu].mean()) if len(iu[0]) else 0.0,
        "max_pairwise_label_jsd": float(jsd[iu].max()) if len(iu[0]) else 0.0,
        "positive_rate_std": float(dist[:, -1].std()),
        "feature_mean_distance": feature_mean_distance(ds),
        "feature_covariance_distance": feature_covariance_distance(ds),
        "sample_count_cv": sample_count_cv(ds),
    }
