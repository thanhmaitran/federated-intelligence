"""Shakespeare loader (LEAF / ``flwrlabs/shakespeare``): next-character prediction.

Each row is an 80-character window ``x`` and the following character ``y``; ``character_id``
(a speaking role) defines the client. Rows of one role are *overlapping sliding windows*, so a
random row split would leak near-identical windows between train and test. We therefore split
every client into **contiguous blocks separated by a gap** of ``window`` rows.
``character_id`` is a literary role, not a real user - results say nothing about real devices.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from federated_intelligence.data.federated_dataset import ClientSplit, FederatedDataset
from federated_intelligence.data.partitioners.natural import contiguous_split_with_gap
from federated_intelligence.utils.seeds import derive_seed

# The 80-symbol alphabet used by LEAF.
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
VOCAB_SIZE = len(ALL_LETTERS)
SEQ_LEN = 80
_LOOKUP = {c: i for i, c in enumerate(ALL_LETTERS)}


def encode(text: str) -> np.ndarray:
    """Map characters to ids; unknown symbols map to the space id (LEAF behaviour)."""
    return np.array([_LOOKUP.get(c, 1) for c in text], dtype=np.int64)


def shakespeare_from_rows(
    xs: Sequence[str],
    ys: Sequence[str],
    character_ids: Sequence[str],
    *,
    seed: int = 42,
    max_clients: int | None = None,
    min_samples_per_client: int = 1000,
    max_samples_per_client: int | None = None,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    global_test_per_client: int = 100,
) -> FederatedDataset:
    cids = np.asarray(character_ids)
    xs, ys = list(xs), list(ys)
    uniq, counts = np.unique(cids, return_counts=True)
    eligible = uniq[counts >= min_samples_per_client]
    rng = np.random.default_rng(derive_seed(seed, "shakespeare"))
    if max_clients is not None and len(eligible) > max_clients:
        eligible = np.sort(rng.choice(eligible, size=max_clients, replace=False))

    clients: dict[str, ClientSplit] = {}
    gx, gy = [], []
    for c in eligible:
        idx = np.flatnonzero(cids == c)  # preserves the original (temporal) order
        if max_samples_per_client is not None:
            idx = idx[:max_samples_per_client]  # first rows = a contiguous passage
        tr, va, te = contiguous_split_with_gap(len(idx), fractions, gap=SEQ_LEN + 1)
        if min(len(tr), len(va), len(te)) == 0:
            continue  # too few windows to form leak-free (gapped) train/val/test blocks
        x = np.stack([encode(xs[i]) for i in idx])
        y = np.array([_LOOKUP.get(ys[i][:1], 1) for i in idx], dtype=np.int64)
        cid = f"role_{c}"
        clients[cid] = ClientSplit(cid, x[tr], y[tr], x[va], y[va], x[te], y[te])
        gx.append(x[te][:global_test_per_client])
        gy.append(y[te][:global_test_per_client])

    if not clients:
        raise ValueError(
            "no role has enough windows for gapped train/val/test blocks "
            f"(need roughly > {int((SEQ_LEN + 1) / min(fractions[1], fractions[2]))} rows per role)"
        )
    meta = {
        "dataset": "shakespeare",
        "partition_method": "natural_character",
        "seed": seed,
        "num_clients": len(clients),
        "split_policy": f"contiguous train/val/test blocks with a gap of {SEQ_LEN + 1} rows",
    }
    return FederatedDataset(
        "shakespeare", clients, np.concatenate(gx), np.concatenate(gy), SEQ_LEN, VOCAB_SIZE, meta
    )


def load_shakespeare(path: str | Path = "data/raw/shakespeare", **kwargs) -> FederatedDataset:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    missing = {"character_id", "x", "y"} - set(ds.column_names)
    if missing:
        raise ValueError(f"Shakespeare is missing fields: {sorted(missing)}")
    return shakespeare_from_rows(ds["x"], ds["y"], ds["character_id"], **kwargs)
