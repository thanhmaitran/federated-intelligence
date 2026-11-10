"""Data quality, partition-consistency and leakage checks (Phase 5.1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from federated_intelligence.data.federated_dataset import FederatedDataset
from federated_intelligence.data.partitioners.base import Partition


@dataclass
class QualityReport:
    stats: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _row_hashes(x: np.ndarray) -> set[bytes]:
    return {hashlib.blake2b(np.ascontiguousarray(r).tobytes(), digest_size=12).digest() for r in x}


def check_dataset(
    ds: FederatedDataset,
    imbalance_warn_ratio: float = 0.05,
    feature_abs_limit: float = 1e3,
) -> QualityReport:
    """Validate a federated dataset. Errors indicate invalid data; warnings are advisory."""
    rep = QualityReport()
    valid_labels = set(range(ds.num_classes))
    n_missing = n_invalid_label = n_dup_within = n_overlap = n_range = 0
    empty: list[str] = []
    sizes: dict[str, int] = {}
    pos_rates: dict[str, float] = {}

    train_hashes: dict[str, set[bytes]] = {}
    for cid, c in ds.clients.items():
        sizes[cid] = c.num_samples
        parts = {
            "train": (c.x_train, c.y_train),
            "val": (c.x_val, c.y_val),
            "test": (c.x_test, c.y_test),
        }
        if min(len(p[1]) for p in parts.values()) == 0:
            empty.append(cid)
        for name, (x, y) in parts.items():
            if len(y) == 0:
                continue
            n_missing += int((~np.isfinite(x)).sum())
            n_invalid_label += int((~np.isin(y, list(valid_labels))).sum())
            n_range += int((np.abs(x) > feature_abs_limit).sum())
            if x.shape[1] != ds.num_features:
                rep.errors.append(f"{cid}/{name}: expected {ds.num_features} features")
        hashes = {n: _row_hashes(p[0]) for n, p in parts.items() if len(p[1])}
        n_dup_within += sum(len(parts[n][1]) - len(h) for n, h in hashes.items())
        n_overlap += len(hashes.get("train", set()) & hashes.get("test", set()))
        n_overlap += len(hashes.get("train", set()) & hashes.get("val", set()))
        train_hashes[cid] = hashes.get("train", set())
        pos_rates[cid] = float(c.y_train.mean()) if len(c.y_train) else float("nan")

    # Cross-client leakage: global test rows must never appear in any client's local data.
    global_hashes = _row_hashes(ds.global_test_x)
    n_global_leak = sum(len(h & global_hashes) for h in train_hashes.values())

    # Global test set sanity.
    if (~np.isfinite(ds.global_test_x)).any():
        rep.errors.append("global test set contains non-finite values")
    if not set(np.unique(ds.global_test_y).tolist()) <= valid_labels:
        rep.errors.append("global test set contains invalid labels")

    counts = np.array(list(sizes.values()))
    classes, class_counts = np.unique(
        np.concatenate([c.y_train for c in ds.clients.values()]), return_counts=True
    )
    minority_share = float(class_counts.min() / class_counts.sum())

    rep.stats = {
        "num_clients": len(ds.clients),
        "total_samples": int(counts.sum()),
        "samples_per_client": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "cv": float(counts.std() / counts.mean()),
        },
        "missing_or_non_finite_values": n_missing,
        "invalid_labels": n_invalid_label,
        "duplicate_rows_within_splits": n_dup_within,
        "train_val_test_overlap_rows": n_overlap,
        "global_test_leakage_rows": n_global_leak,
        "features_out_of_range": n_range,
        "empty_partitions": empty,
        "global_class_distribution": {
            int(c): int(n) for c, n in zip(classes, class_counts, strict=True)
        },
        "minority_class_share": minority_share,
        "client_positive_rate": {
            "min": float(np.nanmin(list(pos_rates.values()))),
            "max": float(np.nanmax(list(pos_rates.values()))),
        },
    }

    if n_missing:
        rep.errors.append(f"{n_missing} missing/non-finite feature values")
    if n_invalid_label:
        rep.errors.append(f"{n_invalid_label} invalid labels")
    if empty:
        rep.errors.append(f"empty splits in clients: {empty[:5]}")
    if n_overlap:
        rep.errors.append(f"{n_overlap} rows shared between train and val/test (leakage)")
    if n_global_leak:
        rep.errors.append(f"{n_global_leak} global-test rows present in client training data")
    if n_dup_within:
        rep.warnings.append(f"{n_dup_within} duplicate rows inside splits")
    if n_range:
        rep.warnings.append(f"{n_range} feature values beyond +/-{feature_abs_limit}")
    if minority_share < imbalance_warn_ratio:
        rep.warnings.append(f"severe class imbalance (minority share {minority_share:.3f})")
    return rep


def check_partition(partition: Partition, num_samples: int) -> QualityReport:
    """Partition consistency: full coverage, disjointness, no empty clients."""
    rep = QualityReport()
    all_idx = (
        np.concatenate([np.asarray(v) for v in partition.values()])
        if partition
        else np.array([], dtype=int)
    )
    unique = np.unique(all_idx)
    empty = [k for k, v in partition.items() if len(v) == 0]
    rep.stats = {
        "num_clients": len(partition),
        "assigned": int(len(all_idx)),
        "unique_assigned": int(len(unique)),
        "expected": int(num_samples),
        "empty_clients": empty,
    }
    if len(all_idx) != len(unique):
        rep.errors.append("partition is not disjoint (index assigned to several clients)")
    if len(unique) != num_samples:
        rep.errors.append(f"partition covers {len(unique)} of {num_samples} samples")
    if len(unique) and (unique.min() < 0 or unique.max() >= num_samples):
        rep.errors.append("partition contains out-of-range indices")
    if empty:
        rep.errors.append(f"empty clients: {empty[:5]}")
    return rep


def check_group_disjoint(train_groups: np.ndarray, test_groups: np.ndarray) -> QualityReport:
    """Writer/character overlap check for unseen-group generalization splits."""
    rep = QualityReport()
    overlap = set(np.unique(train_groups).tolist()) & set(np.unique(test_groups).tolist())
    rep.stats = {"overlapping_groups": sorted(overlap)[:10], "n_overlap": len(overlap)}
    if overlap:
        rep.errors.append(f"{len(overlap)} groups appear in both train and test")
    return rep
