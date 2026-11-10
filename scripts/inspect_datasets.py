#!/usr/bin/env python
"""Re-open the downloaded datasets from local disk and verify schema, size and provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from download_datasets import DATASETS, ROOT, _dir_fingerprint


def main() -> int:
    from datasets import load_from_disk

    rc = 0
    for name, spec in DATASETS.items():
        path = ROOT / "data" / "raw" / name
        print(f"\n{'=' * 60}\nDATASET: {name}\n{'=' * 60}")
        if not path.exists():
            print("NOT FOUND:", path, "- run scripts/download_datasets.py")
            rc = 1
            continue
        ds = load_from_disk(str(path))
        print("samples :", f"{len(ds):,}")
        print("columns :", ds.column_names)
        print("features:", ds.features)
        missing = spec["required"] - set(ds.column_names)
        print("schema  :", "OK" if not missing else f"MISSING {sorted(missing)}")
        clients = len(set(ds[spec["client_key"]]))
        print(f"clients ({spec['client_key']}):", f"{clients:,}")
        meta_path = ROOT / "data" / "metadata" / f"{name}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            same_rows = meta["num_rows"] == len(ds)
            same_fp = meta["fingerprint"] == _dir_fingerprint(path)
            print("provenance:", "OK" if same_rows and same_fp else "MISMATCH (re-download)")
            rc |= int(not (same_rows and same_fp))
        else:
            print(f"provenance: missing data/metadata/{name}.json")
            rc = 1
        rc |= int(bool(missing))
        print("first sample:", {k: (str(v)[:60]) for k, v in ds[0].items()})
    return rc


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
