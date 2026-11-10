#!/usr/bin/env python
"""Load finished experiment runs into the dashboard database (SQLite default, PostgreSQL via DATABASE_URL).

    python scripts/import_results.py
    DATABASE_URL=postgresql+psycopg://fi:pw@localhost/fi python scripts/import_results.py
"""

import argparse
from pathlib import Path

from federated_intelligence.api.db import make_engine, make_session_factory
from federated_intelligence.api.importer import import_results

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "reports/experiment-results")
    ap.add_argument("--database-url", default=None)
    a = ap.parse_args()
    factory = make_session_factory(make_engine(a.database_url))
    with factory() as s:
        print(f"imported {import_results(s, a.results_dir)} runs")
