#!/usr/bin/env python
"""Run an experiment suite.

Examples
--------
    python scripts/run_experiment.py --config configs/experiments/smoke.yaml
    python scripts/run_experiment.py --config configs/experiments/femnist_fedavg.yaml --set federated.num_rounds=20
    python scripts/run_experiment.py --suite configs/experiments/heterogeneity.yaml --smoke
    python scripts/run_experiment.py --suite configs/experiments/heterogeneity.yaml --exp EXP-02 --workers 2
    python scripts/run_experiment.py --suite configs/experiments/privacy.yaml --tracker mlflow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from federated_intelligence.experiments.configuration import ExperimentConfig, _deep_update
from federated_intelligence.experiments.suite import expand_suite, load_suite, run_suite

ROOT = Path(__file__).resolve().parents[1]


def run_single(args: argparse.Namespace) -> None:
    import torch

    from federated_intelligence.experiments.runner import run_experiment
    from federated_intelligence.experiments.tracking import build_tracker

    raw = yaml.safe_load(args.config.read_text()) or {}
    for item in args.set:
        key, _, value = item.partition("=")
        node = raw
        *path, leaf = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    cfg = ExperimentConfig.model_validate(_deep_update({}, raw))
    if args.seeds:
        cfg.experiment.seed = args.seeds[0]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = args.results_dir / ("smoke" if "smoke" in cfg.experiment.name else "single")
    tracker = build_tracker(args.tracker, out, cfg.experiment.name)
    res = run_experiment(cfg, out_dir=out, tracker=tracker, quiet=False)
    f = res["summary"]["final"]
    print(json.dumps({"result": res["result_path"], "test_accuracy": f["test_accuracy"],
                      "test_loss": f["test_loss"], "total_bytes": res["summary"]["total_bytes"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, help="suite YAML (grid of runs over variants and seeds)")
    ap.add_argument("--config", type=Path, help="a single experiment YAML (one run)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. federated.num_rounds=10 (with --config)")
    ap.add_argument("--exp", action="append", help="experiment id(s) to run (default: all in the suite)")
    ap.add_argument("--seeds", type=int, nargs="+", help="override seeds")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "reports" / "experiment-results")
    ap.add_argument("--smoke", action="store_true", help="3 rounds, tiny data, 1 seed (wiring check)")
    ap.add_argument("--tracker", choices=["none", "jsonl", "mlflow"], default="none")
    ap.add_argument("--no-skip", action="store_true", help="re-run finished runs")
    ap.add_argument("--list", action="store_true", help="only list the expanded runs")
    args = ap.parse_args()
    if bool(args.suite) == bool(args.config):
        ap.error("give exactly one of --suite or --config")
    if args.config:
        run_single(args)
        return

    suite = load_suite(args.suite)
    seeds = args.seeds or ([1] if args.smoke else None)
    specs = expand_suite(suite, args.results_dir, args.exp, seeds, args.smoke, args.tracker)
    if args.list:
        for s in specs:
            print(f"{s.exp_id:10s} {s.variant:40s} seed={s.seed}")
        print(f"{len(specs)} runs")
        return
    run_suite(specs, workers=args.workers, skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
