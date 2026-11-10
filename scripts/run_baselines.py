#!/usr/bin/env python
"""Centralized reference baselines on the synthetic dataset (Phase 6).

Trains Logistic Regression and an MLP on the pooled client-train data of the IID-partitioned
and Dirichlet-partitioned populations for several seeds. The reference accuracy of the
federated experiments is the *IID pooled* centralized MLP (same data, model, preprocessing).
Results -> reports/experiment-results/centralized/baselines.json and reports/tables/centralized_baselines.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from federated_intelligence.evaluation.centralized import train_centralized
from federated_intelligence.experiments.configuration import ExperimentConfig
from federated_intelligence.experiments.runner import build_dataset
from federated_intelligence.experiments.suite import load_suite

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(1)
    defaults = load_suite(ROOT / "configs/experiments/heterogeneity.yaml")["defaults"]
    runs, rows = [], []
    for seed in args.seeds:
        cfg = ExperimentConfig.model_validate({**defaults, "experiment": {"name": "central", "seed": seed}})
        ds = build_dataset(cfg)
        for model, hidden in (("logreg", None), ("mlp", [64, 32])):
            r = train_centralized(ds, model, hidden, args.epochs, cfg.training.batch_size,
                                  cfg.training.learning_rate, seed=seed)
            r["dataset"] = "pooled_iid"
            runs.append(r)
            f = r["final"]
            rows.append({"model": model, "seed": seed, "test_accuracy": f["test_accuracy"],
                         "test_macro_f1": f["test_macro_f1"], "test_precision": f["test_macro_precision"],
                         "test_recall": f["test_macro_recall"], "test_roc_auc": f["test_roc_auc"],
                         "test_pr_auc": f["test_pr_auc"], "train_seconds": r["train_seconds"],
                         "num_parameters": r["num_parameters"]})
            print(f"seed={seed} {model:6s} acc={f['test_accuracy']:.4f} auc={f['test_roc_auc']:.4f}")
    out = ROOT / "reports/experiment-results/centralized"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(runs, default=float))
    df = pd.DataFrame(rows)
    (ROOT / "reports/tables").mkdir(parents=True, exist_ok=True)
    summ = df.groupby("model").agg(["mean", "std"]).round(4)
    summ.to_csv(ROOT / "reports/tables/centralized_baselines.csv")
    print(summ[["test_accuracy", "test_macro_f1", "test_roc_auc"]])


if __name__ == "__main__":
    main()
