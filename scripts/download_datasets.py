#!/usr/bin/env python
"""Download FEMNIST and Shakespeare (LEAF) from Hugging Face and record provenance.

    python scripts/download_datasets.py                  # both datasets
    python scripts/download_datasets.py --dataset femnist
    python scripts/download_datasets.py --force          # re-download

Each dataset is saved in Hugging Face Arrow format under ``data/raw/<name>/`` and a provenance
record (source, license, revision, access date, schema, row count, fingerprint) is written to
``data/metadata/<name>.json``. Raw data is git-ignored; only the provenance records are committed.
An existing, valid download is reused (cache); re-running is idempotent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "femnist": {
        "repo": "flwrlabs/femnist",
        "required": {"image", "writer_id", "hsf_id", "character"},
        "license": "BSD-2-Clause",
        "benchmark": "LEAF: A Benchmark for Federated Settings (Caldas et al., 2018)",
        "client_key": "writer_id",
    },
    "shakespeare": {
        "repo": "flwrlabs/shakespeare",
        "required": {"character_id", "x", "y"},
        "license": "BSD-2-Clause",
        "benchmark": "LEAF: A Benchmark for Federated Settings (Caldas et al., 2018)",
        "client_key": "character_id",
    },
}


def _dir_fingerprint(path: Path) -> str:
    """Stable fingerprint of the saved files (names + sizes), cheap for multi-GB datasets."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(path)}:{p.stat().st_size}".encode())
    return h.hexdigest()


def _hub_revision(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception:  # offline or rate-limited: provenance is still recorded, revision unknown
        return None


def is_valid(name: str) -> bool:
    out = ROOT / "data" / "raw" / name
    meta = ROOT / "data" / "metadata" / f"{name}.json"
    if not (out / "dataset_info.json").exists() or not meta.exists():
        return False
    try:
        info = json.loads(meta.read_text())
        return info.get("fingerprint") == _dir_fingerprint(out)
    except (OSError, ValueError):
        return False


def download(name: str, force: bool = False) -> None:
    spec = DATASETS[name]
    out = ROOT / "data" / "raw" / name
    if not force and is_valid(name):
        print(f"[{name}] valid local copy found - skipping download (use --force to refresh)")
        return
    from datasets import load_dataset

    print(f"[{name}] downloading {spec['repo']} ...")
    ds = load_dataset(spec["repo"], split="train")
    missing = spec["required"] - set(ds.column_names)
    if missing:
        raise ValueError(f"[{name}] missing fields: {sorted(missing)}")
    print(ds)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    meta = {
        "name": name,
        "source": f"https://huggingface.co/datasets/{spec['repo']}",
        "hf_revision": _hub_revision(spec["repo"]),
        "original_benchmark": spec["benchmark"],
        "license": spec["license"],
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "num_rows": len(ds),
        "columns": ds.column_names,
        "features": {k: str(v) for k, v in ds.features.items()},
        "splits_available": ["train"],
        "client_key": spec["client_key"],
        "num_clients_in_source": int(len(set(ds[spec["client_key"]]))),
        "fingerprint": _dir_fingerprint(out),
        "preprocessing": "none at download time; see docs/dataset-cards for loader preprocessing",
    }
    meta_dir = ROOT / "data" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] saved to {out} ({len(ds):,} rows); provenance -> data/metadata/{name}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    rc = 0
    for n in names:
        try:
            download(n, args.force)
        except Exception as exc:  # network / auth problems should not hide the other dataset
            print(f"[{n}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
