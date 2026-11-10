"""FEMNIST loader (LEAF / ``flwrlabs/femnist``): natural partition by ``writer_id``.

Two evaluation protocols are produced and must be reported separately:

* **Held-out samples of participating writers** - each client keeps a private test split
  (``ClientSplit.x_test``); measures personalisation / within-client generalisation.
* **Unseen-writer generalisation** - a set of writers that never takes part in training; their
  samples form ``global_test``. No writer appears in both sets (checked by
  :func:`check_group_disjoint`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from federated_intelligence.data.federated_dataset import ClientSplit, FederatedDataset, split_indices
from federated_intelligence.data.partitioners.natural import split_groups
from federated_intelligence.data.validators import check_group_disjoint
from federated_intelligence.utils.seeds import derive_seed

REQUIRED_COLUMNS = {"image", "writer_id", "hsf_id", "character"}
NUM_CLASSES = 62


def _encode_labels(chars: np.ndarray) -> np.ndarray:
    if np.issubdtype(chars.dtype, np.integer):
        return chars.astype(np.int64)
    classes = sorted({str(c) for c in chars})
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[str(c)] for c in chars], dtype=np.int64)


def femnist_from_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    writers: np.ndarray,
    *,
    seed: int = 42,
    holdout_writer_fraction: float = 0.1,
    max_clients: int | None = None,
    min_samples_per_client: int = 30,
    max_samples_per_client: int | None = None,
    unseen_test_cap: int = 5000,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> FederatedDataset:
    """Build a :class:`FederatedDataset` from decoded arrays (``images``: (N, 28, 28) uint8/float)."""
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[:, None, :, :]
    x_all = (images.astype(np.float32) / 255.0) if images.dtype == np.uint8 else images.astype(np.float32)
    y_all = _encode_labels(np.asarray(labels))
    writers = np.asarray(writers)

    uniq, counts = np.unique(writers, return_counts=True)
    eligible = uniq[counts >= min_samples_per_client]
    seen_writers, unseen_writers = split_groups(eligible, holdout_writer_fraction, derive_seed(seed, "writers"))
    rng = np.random.default_rng(derive_seed(seed, "femnist"))
    if max_clients is not None and len(seen_writers) > max_clients:
        seen_writers = np.sort(rng.choice(seen_writers, size=max_clients, replace=False))

    clients: dict[str, ClientSplit] = {}
    for w in seen_writers:
        idx = np.flatnonzero(writers == w)
        if max_samples_per_client is not None and len(idx) > max_samples_per_client:
            idx = np.sort(rng.choice(idx, size=max_samples_per_client, replace=False))
        tr, va, te = split_indices(len(idx), fractions, np.random.default_rng(derive_seed(seed, "split", str(w))))
        cid = f"writer_{w}"
        clients[cid] = ClientSplit(cid, x_all[idx[tr]], y_all[idx[tr]], x_all[idx[va]], y_all[idx[va]],
                                   x_all[idx[te]], y_all[idx[te]])

    unseen_idx = np.flatnonzero(np.isin(writers, unseen_writers))
    if len(unseen_idx) > unseen_test_cap:
        unseen_idx = np.sort(rng.choice(unseen_idx, size=unseen_test_cap, replace=False))
    report = check_group_disjoint(seen_writers, unseen_writers)
    if not report.passed:
        raise ValueError(f"writer leakage between seen and unseen sets: {report.errors}")

    meta = {
        "dataset": "femnist",
        "partition_method": "natural_writer",
        "seed": seed,
        "num_clients": len(clients),
        "num_unseen_writers": int(len(unseen_writers)),
        "evaluation": {
            "client_test": "held-out samples of participating writers",
            "global_test": "samples of writers never used in training (unseen-writer generalisation)",
        },
        "writer_disjoint_check": report.passed,
    }
    return FederatedDataset("femnist", clients, x_all[unseen_idx], y_all[unseen_idx], 28 * 28, NUM_CLASSES, meta)


def load_femnist(path: str | Path = "data/raw/femnist", max_writers: int | None = None, **kwargs) -> FederatedDataset:
    """Load the Hugging Face Arrow dataset saved by ``scripts/download_datasets.py``."""
    from datasets import load_from_disk  # local import: optional dependency

    ds = load_from_disk(str(path))
    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(f"FEMNIST is missing fields: {sorted(missing)}")
    writers_all = np.asarray(ds["writer_id"])
    if max_writers is not None:  # decode only the rows we need (images are the expensive column)
        rng = np.random.default_rng(derive_seed(kwargs.get("seed", 42), "pick_writers"))
        chosen = rng.choice(np.unique(writers_all), size=min(max_writers, len(np.unique(writers_all))), replace=False)
        rows = np.flatnonzero(np.isin(writers_all, chosen))
        ds = ds.select(rows)
        writers_all = writers_all[rows]
    images = np.stack([np.asarray(im, dtype=np.uint8) for im in ds["image"]])
    return femnist_from_arrays(images, np.asarray(ds["character"]), writers_all, **kwargs)
