#!/bin/sh
# Import finished experiment runs into the database, then serve the API.
set -e
if [ "${FI_IMPORT_RESULTS:-1}" = "1" ]; then
  python scripts/import_results.py --results-dir "${FI_RESULTS_DIR:-reports/experiment-results}"
fi
exec uvicorn federated_intelligence.api.main:app --host 0.0.0.0 --port 8000
