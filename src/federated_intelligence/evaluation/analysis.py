"""Aggregation of persisted experiment runs into analysis-ready tables.

Every run is one JSON file (see ``experiments/runner.py``). This module flattens them into

* ``runs``   - one row per run (config axes + final metrics + cost metrics), and
* ``curves`` - one row per (run, evaluated round),

and provides the statistics used in the report: mean +- SD, Student-t 95% CI, paired
comparisons across seeds (the two arms share data, partition and initial weights for a given
seed) with Holm correction, and convergence descriptors.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from federated_intelligence.evaluation.convergence import area_under_curve, curve_volatility
from federated_intelligence.evaluation.statistical_tests import compare_paired, summarize

SKIP_DIRS = {"smoke", "single", "centralized"}


def iter_result_files(results_dir: str | Path, prefix: str | None = "EXP-") -> Iterable[Path]:
    root = Path(results_dir)
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in SKIP_DIRS or d.name.startswith("_"):
            continue
        if prefix and not d.name.startswith(prefix):
            continue
        yield from sorted(d.glob("*.json"))


def run_row(data: dict) -> dict:
    cfg, summ = data["config"], data["summary"]
    fin = summ["final"]
    priv = summ.get("privacy") or {}
    hist = [h for h in data["history"] if "test_accuracy" in h and h["round"] > 0]
    rounds = [h["round"] for h in hist]
    acc = [h["test_accuracy"] for h in hist]
    row = {
        "exp_id": data.get("exp_id", ""),
        "variant": data.get("variant", data["name"]),
        "seed": data["seed"],
        "research_question": data.get("research_question", ""),
        **{f"axis_{k}": v for k, v in (data.get("axes") or {}).items()},
        "algorithm": cfg["federated"]["algorithm"],
        "mu": cfg["federated"]["mu"],
        "participation_rate": cfg["federated"]["participation_rate"],
        "dropout_rate": cfg["federated"]["dropout_rate"],
        "num_rounds": cfg["federated"]["num_rounds"],
        "selection": cfg["federated"]["selection"],
        "model": cfg["model"]["name"],
        "partition": cfg["dataset"]["partition"] if cfg["dataset"]["mode"] == "pooled" else "native",
        "scenario": cfg["dataset"]["scenario"] if cfg["dataset"]["mode"] == "native" else "",
        "alpha": cfg["dataset"]["alpha"] if cfg["dataset"]["partition"] == "dirichlet" else np.nan,
        "num_clients": cfg["dataset"]["num_clients"],
        "compressor": cfg["communication"]["compressor"],
        "topk_ratio": cfg["communication"]["topk_ratio"] if cfg["communication"]["compressor"] == "topk" else np.nan,
        "error_feedback": cfg["communication"]["error_feedback"],
        "downlink": cfg["communication"]["downlink"],
        "dp_enabled": cfg["privacy"]["enabled"],
        "clip_norm": cfg["privacy"]["clip_norm"] if cfg["privacy"]["enabled"] else np.nan,
        "noise_multiplier": priv.get("noise_multiplier", np.nan),
        "epsilon": priv.get("epsilon", np.nan) if cfg["privacy"]["enabled"] else np.inf,
        "delta": priv.get("delta", np.nan),
        "network": cfg["network"]["enabled"],
        "deadline_s": cfg["network"]["deadline_s"],
        "test_accuracy": fin["test_accuracy"],
        "test_macro_f1": fin["test_macro_f1"],
        "test_loss": fin["test_loss"],
        "test_roc_auc": fin.get("test_roc_auc", np.nan),
        "test_pr_auc": fin.get("test_pr_auc", np.nan),
        "val_accuracy": fin["val_accuracy"],
        "client_acc_mean": fin.get("client_acc_mean", np.nan),
        "client_acc_std": fin.get("client_acc_std", np.nan),
        "client_acc_min": fin.get("client_acc_min", np.nan),
        "client_acc_p10": fin.get("client_acc_p10", np.nan),
        "rounds_to_target": summ.get("rounds_to_target_val_accuracy"),
        "auc_accuracy_curve": area_under_curve(rounds, acc) if len(rounds) > 1 else np.nan,
        "volatility_last10": curve_volatility(acc, 10),
        "total_uplink_bytes": summ["total_uplink_bytes"],
        "total_downlink_bytes": summ["total_downlink_bytes"],
        "total_bytes": summ["total_bytes"],
        "mean_uplink_bytes_per_update": summ["mean_uplink_bytes_per_update"],
        "uplink_compression_ratio": summ["uplink_compression_ratio"],
        "participating_updates": summ["total_participating_updates"],
        "skipped_rounds": summ["skipped_rounds"],
        "simulated_seconds": summ["simulated_seconds_total"],
        "compute_seconds": summ["compute_seconds_total"],
        "wall_seconds": summ["wall_seconds"],
        "executor": summ["executor"],
    }
    return row


def load_runs(results_dir: str | Path, prefix: str | None = "EXP-") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for f in iter_result_files(results_dir, prefix):
        data = json.loads(f.read_text())
        if "exp_id" not in data:
            continue
        rows.append(run_row(data))
        for h in data["history"]:
            if "test_accuracy" not in h:
                continue
            curves.append(
                {
                    "exp_id": data["exp_id"], "variant": data["variant"], "seed": data["seed"],
                    "round": h["round"], "test_accuracy": h["test_accuracy"], "test_loss": h["test_loss"],
                    "test_macro_f1": h["test_macro_f1"], "epsilon": h.get("epsilon", np.nan),
                    "cum_bytes": np.nan, "participants": h.get("participants", np.nan),
                }
            )
        # cumulative bytes along the curve
        cum = 0
        per_round = {h["round"]: h.get("uplink_bytes", 0) + h.get("downlink_bytes", 0) for h in data["history"]}
        for c in curves[-sum(1 for h in data["history"] if "test_accuracy" in h):]:
            cum = sum(v for r, v in per_round.items() if r <= c["round"])
            c["cum_bytes"] = cum
    return pd.DataFrame(rows), pd.DataFrame(curves)


def mean_ci_table(df: pd.DataFrame, by: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean, SD and 95% t-CI of each metric per group."""
    out = []
    for key, g in df.groupby(list(by), dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(by, key, strict=True))
        rec["n"] = len(g)
        for m in metrics:
            vals = g[m].dropna().to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                rec[f"{m}_mean"] = rec[f"{m}_sd"] = rec[f"{m}_ci_low"] = rec[f"{m}_ci_high"] = np.nan
                continue
            s = summarize(vals)
            rec[f"{m}_mean"], rec[f"{m}_sd"], rec[f"{m}_ci_low"], rec[f"{m}_ci_high"] = s.mean, s.sd, s.ci_low, s.ci_high
        out.append(rec)
    return pd.DataFrame(out)


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs are passed through)."""
    p = np.asarray(pvals, float)
    adj = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def paired_table(
    df: pd.DataFrame, group_cols: Sequence[str], arm_col: str, a: str, b: str, metric: str = "test_accuracy"
) -> pd.DataFrame:
    """Paired comparison of arm ``a`` vs ``b`` (matched by seed) inside each group, Holm-adjusted."""
    rows = []
    for key, g in df.groupby(list(group_cols), dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        pa = g[g[arm_col] == a].set_index("seed")[metric]
        pb = g[g[arm_col] == b].set_index("seed")[metric]
        seeds = sorted(set(pa.index) & set(pb.index))
        if len(seeds) < 2:
            continue
        r = compare_paired(pa.loc[seeds].to_numpy(), pb.loc[seeds].to_numpy())
        rows.append({**dict(zip(group_cols, key, strict=True)), "a": a, "b": b, "metric": metric,
                     "n": r["n"], "mean_a": float(pa.loc[seeds].mean()), "mean_b": float(pb.loc[seeds].mean()),
                     "mean_diff": r["mean_diff"], "ci_low": r["diff_ci95"][0], "ci_high": r["diff_ci95"][1],
                     "cohens_dz": r["cohens_dz"], "p_value": r.get("p_value", np.nan),
                     "wilcoxon_p": r.get("wilcoxon_p", np.nan)})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm_adjust(t["p_value"].tolist())
        t["significant_holm_0.05"] = t["p_holm"] < 0.05
    return t


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Dependency-free GitHub markdown table."""
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isinf(v):
                return "inf"
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
