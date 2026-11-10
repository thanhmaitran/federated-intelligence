"""Read back a synthetic dataset written by ``TabularGenerator.write`` and verify checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from federated_intelligence.data.federated_dataset import ClientSplit, FederatedDataset


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_synthetic(directory: str | Path, verify: bool = True) -> FederatedDataset:
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text())
    if verify:
        for name, info in manifest["files"].items():
            if _sha256(d / name) != info["sha256"]:
                raise ValueError(f"checksum mismatch for {d / name}")
    df = pd.read_parquet(d / "clients.parquet")
    cols = [c for c in df.columns if c.startswith("f")]
    clients: dict[str, ClientSplit] = {}
    for cid, g in df.groupby("client_id", sort=True):
        parts = {s: g[g["split"] == s] for s in ("train", "val", "test")}
        clients[str(cid)] = ClientSplit(
            str(cid),
            *[
                arr
                for s in ("train", "val", "test")
                for arr in (
                    parts[s][cols].to_numpy("float32"),
                    parts[s]["label"].to_numpy("int64"),
                )
            ],
        )
    gdf = pd.read_parquet(d / "global_test.parquet")
    return FederatedDataset(
        manifest["dataset"],
        clients,
        gdf[cols].to_numpy("float32"),
        gdf["label"].to_numpy("int64"),
        manifest["num_features"],
        manifest["num_classes"],
        {k: v for k, v in manifest.items() if k not in ("files", "client_parameters")},
    )
