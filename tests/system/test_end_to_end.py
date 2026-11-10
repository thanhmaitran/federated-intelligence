"""End-to-end: CLI -> dataset -> training -> persisted result -> analysis."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_cli_single_config_end_to_end(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config",
         str(ROOT / "configs/experiments/smoke.yaml"), "--results-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    res = json.loads(files[0].read_text())
    assert res["summary"]["final"]["test_accuracy"] > 0.5
    assert res["summary"]["total_bytes"] > 0
    assert len(res["history"]) == 4  # round 0 + 3 rounds
    assert res["environment"]["python"]
    assert res["config_hash"]


@pytest.mark.slow
def test_suite_smoke_expansion_and_resume(tmp_path):
    from federated_intelligence.experiments.suite import expand_suite, load_suite, run_suite

    suite = load_suite(ROOT / "configs/experiments/privacy.yaml")
    specs = expand_suite(suite, tmp_path, ["EXP-04"], seeds=[1], smoke=True)
    specs = [s for s in specs if s.variant in ("nodp", "eps10")]
    assert len(specs) == 2
    run_suite(specs, workers=1, progress=False)
    assert all(s.result_path.exists() for s in specs)
    mtime = specs[0].result_path.stat().st_mtime_ns
    run_suite(specs, workers=1, progress=False)  # resume: nothing re-executed
    assert specs[0].result_path.stat().st_mtime_ns == mtime
    dp = json.loads(next(s for s in specs if s.variant == "eps10").result_path.read_text())
    assert dp["summary"]["privacy"]["epsilon"] <= 10.0 + 1e-6
    assert dp["axes"]["variant"] == "eps10"


def test_every_suite_config_validates():
    from federated_intelligence.experiments.suite import expand_suite, load_suite

    for name in ("baseline", "heterogeneity", "privacy", "communication", "pilot"):
        suite = load_suite(ROOT / f"configs/experiments/{name}.yaml")
        specs = expand_suite(suite, "/tmp/unused")
        assert specs, name
