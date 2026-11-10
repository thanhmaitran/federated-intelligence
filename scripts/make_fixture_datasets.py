#!/usr/bin/env python
"""Create tiny FEMNIST/Shakespeare-shaped *fixture* datasets under data/fixtures/ (toy data!).

    python scripts/make_fixture_datasets.py
    python scripts/run_experiment.py --config configs/experiments/fixture_femnist_smoke.yaml
"""

from pathlib import Path

from federated_intelligence.data.loaders.fixtures import make_femnist_fixture, make_shakespeare_fixture

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(make_femnist_fixture(ROOT / "data" / "fixtures" / "femnist"))
    print(make_shakespeare_fixture(ROOT / "data" / "fixtures" / "shakespeare"))
    print("Fixture data is random toy data: use it for wiring checks only, never for reported results.")
