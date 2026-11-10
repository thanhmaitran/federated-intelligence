#!/usr/bin/env bash
# Run every experiment suite (smoke -> full). Resumable: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-2}"
for suite in baseline heterogeneity privacy communication; do
  python scripts/run_experiment.py --suite "configs/experiments/${suite}.yaml" --workers "$WORKERS" "$@"
done
python scripts/run_baselines.py
