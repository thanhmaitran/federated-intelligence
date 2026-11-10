#!/usr/bin/env python
"""Validate generated datasets and (if present) FEMNIST/Shakespeare - schema, leakage, partitions.

Exit code 1 if any check fails. Use in CI after ``generate_synthetic.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from federated_intelligence.data.loaders import load_synthetic
from federated_intelligence.data.validators import check_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rc = 0
    base = ROOT / "data" / "synthetic"
    dirs = sorted(p for p in base.glob("*") if (p / "manifest.json").exists()) if base.exists() else []
    if not dirs:
        print("no synthetic datasets found - run scripts/generate_synthetic.py first")
        return 1
    for d in dirs:
        ds = load_synthetic(d, verify=True)  # also verifies SHA-256 checksums
        rep = check_dataset(ds)
        print(f"{d.name:20s} {'PASS' if rep.passed else 'FAIL'}  warnings={len(rep.warnings)}")
        for e in rep.errors:
            print("   ERROR:", e)
        rc |= int(not rep.passed)
    raw = ROOT / "data" / "raw"
    for name in ("femnist", "shakespeare"):
        if (raw / name / "dataset_info.json").exists():
            print(f"{name}: raw data present (see scripts/inspect_datasets.py for schema checks)")
        else:
            print(f"{name}: not downloaded (skipped)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
