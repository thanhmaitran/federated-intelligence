PYTHON ?= python

.PHONY: install lint typecheck test test-fast coverage smoke data profile baselines experiments report api clean

install:
	$(PYTHON) -m pip install -e ".[dev,api]"

lint:
	ruff check src tests scripts

typecheck:
	mypy src/federated_intelligence

test:
	pytest

test-fast:
	pytest -m "not slow and not ray"

coverage:
	pytest --cov=federated_intelligence --cov-report=term-missing -m "not ray"

smoke:
	$(PYTHON) scripts/run_experiment.py --config configs/experiments/smoke.yaml

data:
	$(PYTHON) scripts/generate_synthetic.py

profile:
	$(PYTHON) scripts/validate_data.py

baselines:
	$(PYTHON) scripts/run_baselines.py

experiments:
	$(PYTHON) scripts/run_suite.py --suite all

report:
	$(PYTHON) scripts/generate_report.py

api:
	uvicorn federated_intelligence.api.main:app --reload

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
