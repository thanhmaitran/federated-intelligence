"""Partition manifests: reproducible, checksummed descriptions of a partition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from federated_intelligence.data.partitioners.base import (
    PARTITIONER_VERSION,
    BasePartitioner,
    Partition,
)


def partition_checksum(partition: Partition) -> str:
    h = hashlib.sha256()
    for cid in sorted(partition):
        h.update(cid.encode())
        h.update(np.asarray(partition[cid], dtype=np.int64).tobytes())
    return h.hexdigest()


def build_manifest(
    partition_id: str,
    dataset: str,
    partitioner: BasePartitioner,
    partition: Partition,
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    include_indices: bool = True,
) -> dict:
    clients = {}
    for cid, idx in sorted(partition.items()):
        n = len(idx)
        n_train = int(round(n * split_fractions[0]))
        n_val = int(round(n * split_fractions[1]))
        entry = {
            "num_samples": n,
            "train_samples": n_train,
            "validation_samples": n_val,
            "test_samples": n - n_train - n_val,
        }
        if include_indices:
            entry["indices"] = idx.tolist()
        clients[cid] = entry
    return {
        "partition_id": partition_id,
        "dataset": dataset,
        "num_clients": len(partition),
        "partition_method": partitioner.method,
        **partitioner.params(),
        "seed": partitioner.seed,
        "partitioner_version": PARTITIONER_VERSION,
        "split_policy": {
            "train": split_fractions[0],
            "validation": split_fractions[1],
            "test": split_fractions[2],
        },
        "checksum_sha256": partition_checksum(partition),
        "clients": clients,
    }


def save_manifest(manifest: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest['partition_id']}.json"
    path.write_text(json.dumps(manifest))
    return path
