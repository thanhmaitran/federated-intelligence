#!/usr/bin/env python
"""Aggregate every finished run into tables (CSV + markdown), statistics and figures.

    python scripts/generate_report.py

Outputs
-------
reports/tables/*.csv, reports/tables/results_summary.md   mean +- SD, 95% CI, paired tests (Holm)
reports/figures/experiments/*.png + captions.json         figures with a description each
docs/images/experiments|communication|datasets/*.png      curated copies used by README/report

Nothing here invents numbers: every value is computed from reports/experiment-results/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from federated_intelligence.evaluation.analysis import (
    load_runs,
    mean_ci_table,
    paired_table,
    to_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "experiment-results"
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures" / "experiments"
DOCS = ROOT / "docs" / "images"

# Validated categorical palette (blue, orange, aqua, yellow, magenta, green, violet), fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "regular", "legend.frameon": False, "lines.linewidth": 2,
})

CAPTIONS: dict[str, str] = {}
COPIES: dict[str, str] = {}  # figure file -> docs/images/<subdir>/<name>


def save(fig, name: str, caption: str, copy_to: str | None = None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    CAPTIONS[name] = caption
    if copy_to:
        COPIES[name] = copy_to


def band(ax, curves: pd.DataFrame, color, label, metric="test_accuracy", x="round"):
    g = curves.groupby(x)[metric]
    m, s, n = g.mean(), g.std(ddof=1), g.count()
    from scipy import stats

    half = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * s / np.sqrt(n)
    ax.plot(m.index, m.values, color=color, label=label)
    ax.fill_between(m.index, (m - half).values, (m + half).values, color=color, alpha=0.18, linewidth=0)


def md_section(title: str, table: pd.DataFrame, note: str = "") -> str:
    return f"### {title}\n\n{note}\n\n{to_markdown(table)}\n\n" if len(table) else ""


def fmt_ms(df: pd.DataFrame, metric: str, digits: int = 4) -> pd.Series:
    return df[f"{metric}_mean"].round(digits).astype(str) + " ± " + df[f"{metric}_sd"].round(digits).astype(str)


def centralized_reference() -> dict[str, float]:
    p = RES / "centralized" / "baselines.json"
    if not p.exists():
        return {}
    runs = json.loads(p.read_text())
    out = {}
    for m in {r["model"] for r in runs}:
        out[m] = float(np.mean([r["final"]["test_accuracy"] for r in runs if r["model"] == m]))
    return out


def main() -> None:
    runs, curves = load_runs(RES)
    if runs.empty:
        raise SystemExit("no finished runs found")
    TAB.mkdir(parents=True, exist_ok=True)
    runs.to_csv(TAB / "all_runs.csv", index=False)
    ref = centralized_reference()
    md = ["# Auto-generated results summary\n",
          "All numbers are computed from `reports/experiment-results/` by `scripts/generate_report.py`. "
          "`n` is the number of independent seeds; ± is the sample SD; CI = 95% Student-t. "
          "Paired tests compare arms on the same seeds (same data, partition and initial weights) and are "
          "Holm-corrected within each family.\n"]
    if ref:
        md.append("Centralized reference (pooled training data, same model/optimiser): "
                  + ", ".join(f"{k} = {v:.4f}" for k, v in ref.items()) + " test accuracy.\n")
    M = ["test_accuracy", "test_macro_f1", "test_roc_auc", "client_acc_mean", "client_acc_min"]

    def cur(exp):
        return curves[curves.exp_id == exp]

    def rr(exp):
        return runs[runs.exp_id == exp]

    # ------------------------------------------------------------------ EXP-00 / EXP-01
    e0, e1 = rr("EXP-00"), rr("EXP-01")
    if len(e1):
        t = mean_ci_table(pd.concat([e0.assign(variant="iid_ref") if False else e1]), ["variant"],
                          M + ["rounds_to_target", "auc_accuracy_curve", "volatility_last10"])
        t.to_csv(TAB / "exp01_iid_vs_noniid.csv", index=False)
        pt = paired_table(e1.assign(g="all"), ["g"], "variant", "iid", "dirichlet0.5")
        pt.to_csv(TAB / "exp01_paired.csv", index=False)
        view = pd.DataFrame({"partition": t["variant"], "n": t["n"], "test acc": fmt_ms(t, "test_accuracy"),
                             "macro-F1": fmt_ms(t, "test_macro_f1"), "worst-client acc": fmt_ms(t, "client_acc_min"),
                             "rounds→0.80 val acc": t["rounds_to_target_mean"].round(1),
                             "acc-curve AUC": t["auc_accuracy_curve_mean"].round(4)})
        md.append(md_section("EXP-01 – IID vs Dirichlet(0.5), FedAvg", view,
                             "Same population, same seeds; the only difference is the partition."))
        if len(pt):
            md.append(md_section("EXP-01 – paired IID − Dirichlet(0.5) (test accuracy)", pt.round(4)))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for i, (v, lab) in enumerate([("iid", "IID"), ("dirichlet0.5", "Dirichlet α = 0.5")]):  # noqa: B007
            band(ax, cur("EXP-01").query("variant == @v"), C[i], lab)
        if "mlp" in ref:
            ax.axhline(ref["mlp"], color=INK2, ls="--", lw=1)
            ax.text(1, ref["mlp"] + 0.003, "centralized MLP", color=INK2, fontsize=8)
        ax.set(xlabel="communication round", ylabel="global test accuracy", title="FedAvg convergence: IID vs non-IID",
               ylim=(0.5, 0.95))
        ax.legend(loc="lower right")
        save(fig, "exp01_convergence_iid_vs_noniid.png",
             "Global test accuracy versus round for FedAvg on an IID and a Dirichlet(0.5) partition of the same "
             "population (mean and 95% CI over seeds). The dashed line is the centralized MLP reference.",
             "experiments/convergence-comparison.png")

    # ------------------------------------------------------------------ EXP-01B
    e1b = rr("EXP-01B")
    if len(e1b):
        het = pd.read_csv(TAB / "heterogeneity_by_scenario.csv") if (TAB / "heterogeneity_by_scenario.csv").exists() else None
        t = mean_ci_table(e1b, ["axis_scenario"], M + ["rounds_to_target"])
        if het is not None:
            h = het.assign(axis_scenario=het["dataset"].str.replace("scenario:", "", regex=False))
            t = t.merge(h[["axis_scenario", "mean_pairwise_label_jsd", "feature_mean_distance", "sample_count_cv"]],
                        on="axis_scenario", how="left")
        t.to_csv(TAB / "exp01b_scenarios.csv", index=False)
        order = ["iid", "label_skew", "feature_skew", "quantity_skew", "combined_non_iid"]
        t = t.set_index("axis_scenario").loc[[o for o in order if o in set(t["axis_scenario"])]].reset_index()
        view = pd.DataFrame({"scenario": t["axis_scenario"], "n": t["n"], "test acc": fmt_ms(t, "test_accuracy"),
                             "worst-client acc": fmt_ms(t, "client_acc_min")})
        for col, nm in [("mean_pairwise_label_jsd", "label JSD"), ("feature_mean_distance", "feature-mean dist"),
                        ("sample_count_cv", "sample CV")]:
            if col in t:
                view[nm] = t[col].round(3)
        md.append(md_section("EXP-01B – one heterogeneity source at a time (native generator)", view,
                             "Heterogeneity statistics are measured on seed-42 data (reports/tables/heterogeneity_by_scenario.csv)."))
        pairs = []
        base = e1b[e1b.axis_scenario == "iid"].set_index("seed")["test_accuracy"]
        for sc in order[1:]:
            o = e1b[e1b.axis_scenario == sc].set_index("seed")["test_accuracy"]
            sd = sorted(set(base.index) & set(o.index))
            if len(sd) > 1:
                from federated_intelligence.evaluation.statistical_tests import compare_paired
                r = compare_paired(o.loc[sd], base.loc[sd])
                pairs.append({"scenario": sc, "n": r["n"], "mean_diff_vs_iid": r["mean_diff"],
                              "ci_low": r["diff_ci95"][0], "ci_high": r["diff_ci95"][1], "d_z": r["cohens_dz"],
                              "p": r.get("p_value", np.nan)})
        if pairs:
            from federated_intelligence.evaluation.analysis import holm_adjust
            pp = pd.DataFrame(pairs); pp["p_holm"] = holm_adjust(pp["p"].tolist())
            pp.to_csv(TAB / "exp01b_paired.csv", index=False)
            md.append(md_section("EXP-01B – paired difference to the IID scenario (test accuracy)", pp.round(4),
                                 "Caveat: each scenario has its own data-generating process, so this contrasts scenarios, not a single treatment."))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        xs = np.arange(len(t))
        ax.bar(xs, t["test_accuracy_mean"], color=C[0], width=0.6,
               yerr=[t["test_accuracy_mean"] - t["test_accuracy_ci_low"], t["test_accuracy_ci_high"] - t["test_accuracy_mean"]],
               error_kw={"ecolor": INK2, "capsize": 3})
        ax.set_xticks(xs, [s.replace("_", "\n") for s in t["axis_scenario"]])
        lo = float(t["test_accuracy_ci_low"].min())
        ax.set(ylim=(max(0.5, lo - 0.05), min(1.0, float(t["test_accuracy_ci_high"].max()) + 0.03)),
               ylabel="global test accuracy", title="FedAvg by heterogeneity source (95% CI)")
        save(fig, "exp01b_heterogeneity_sources.png",
             "Final global test accuracy of FedAvg for each isolated source of heterogeneity from the synthetic "
             "generator (label, feature and quantity skew, and all three combined); bars show the mean and 95% CI over seeds.",
             "experiments/heterogeneity-sources.png")

    # ------------------------------------------------------------------ EXP-02 / 02B
    e2 = rr("EXP-02")
    if len(e2):
        t = mean_ci_table(e2, ["alpha", "algorithm"], M + ["rounds_to_target", "auc_accuracy_curve", "volatility_last10"])
        t = t.sort_values(["alpha", "algorithm"], ascending=[False, True])
        t.to_csv(TAB / "exp02_alpha_algorithm.csv", index=False)
        pt = paired_table(e2, ["alpha"], "algorithm", "fedprox", "fedavg")
        pt2 = paired_table(e2, ["alpha"], "algorithm", "fedprox", "fedavg", metric="auc_accuracy_curve")
        pt.to_csv(TAB / "exp02_paired_final_accuracy.csv", index=False)
        pt2.to_csv(TAB / "exp02_paired_curve_auc.csv", index=False)
        view = pd.DataFrame({"alpha": t["alpha"], "algorithm": t["algorithm"], "n": t["n"],
                             "test acc": fmt_ms(t, "test_accuracy"), "worst-client acc": fmt_ms(t, "client_acc_min"),
                             "rounds→0.80": t["rounds_to_target_mean"].round(1),
                             "curve AUC": t["auc_accuracy_curve_mean"].round(4),
                             "volatility": t["volatility_last10_mean"].round(4)})
        md.append(md_section("EXP-02 – Dirichlet α × algorithm (FedProx μ = 1.0)", view))
        md.append(md_section("EXP-02 – paired FedProx − FedAvg, final test accuracy", pt.round(4),
                             "Holm-corrected across the four α values."))
        md.append(md_section("EXP-02 – paired FedProx − FedAvg, accuracy-curve AUC (convergence speed)", pt2.round(4)))
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
        ax = axes[0]
        for i, (alg, lab) in enumerate([("fedavg", "FedAvg"), ("fedprox", "FedProx (μ = 1)")]):
            s = t[t.algorithm == alg].sort_values("alpha")
            ax.errorbar(s["alpha"], s["test_accuracy_mean"], yerr=[s["test_accuracy_mean"] - s["test_accuracy_ci_low"],
                        s["test_accuracy_ci_high"] - s["test_accuracy_mean"]], color=C[i], marker="o", ms=6, capsize=3, label=lab)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set(xlabel="Dirichlet α (←more heterogeneous)", ylabel="final global test accuracy", title="Accuracy vs heterogeneity")
        ax.legend()
        ax = axes[1]
        c2 = cur("EXP-02")
        for i, (alg, lab) in enumerate([("fedavg", "FedAvg"), ("fedprox", "FedProx (μ = 1)")]):
            band(ax, c2[c2.variant == f"alpha0.1__{alg}"], C[i], lab)
        ax.set(xlabel="communication round", ylabel="global test accuracy", title="Convergence at α = 0.1")
        ax.legend(loc="lower right")
        save(fig, "exp02_fedavg_vs_fedprox.png",
             "Left: final test accuracy of FedAvg and FedProx (μ = 1) for four Dirichlet concentrations (mean and 95% CI "
             "over seeds). Right: convergence curves for the most heterogeneous partition (α = 0.1).",
             "experiments/fedavg-vs-fedprox.png")
    e2b = rr("EXP-02B")
    if len(e2b):
        e2b = e2b.assign(mu_eff=e2b["mu"])
        t = mean_ci_table(e2b, ["alpha", "mu_eff"], ["test_accuracy", "auc_accuracy_curve", "volatility_last10"])
        t.to_csv(TAB / "exp02b_mu_sweep.csv", index=False)
        md.append(md_section("EXP-02B – FedProx μ sensitivity (μ = 0 is FedAvg)", pd.DataFrame({
            "alpha": t["alpha"], "mu": t["mu_eff"], "n": t["n"], "test acc": fmt_ms(t, "test_accuracy"),
            "curve AUC": t["auc_accuracy_curve_mean"].round(4), "volatility": t["volatility_last10_mean"].round(4)})))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for i, a in enumerate(sorted(t["alpha"].unique())):
            s = t[t.alpha == a].sort_values("mu_eff")
            x = np.where(s["mu_eff"] == 0, 0.003, s["mu_eff"])
            ax.errorbar(x, s["test_accuracy_mean"], yerr=[s["test_accuracy_mean"] - s["test_accuracy_ci_low"],
                        s["test_accuracy_ci_high"] - s["test_accuracy_mean"]], color=C[i], marker="o", capsize=3, label=f"α = {a}")
        ax.set_xscale("log"); ax.set(xlabel="proximal coefficient μ (leftmost point = 0, FedAvg)", ylabel="final global test accuracy",
                                     title="FedProx: sensitivity to μ")
        ax.legend()
        save(fig, "exp02b_mu_sensitivity.png",
             "Final test accuracy as a function of the FedProx proximal coefficient μ for α = 0.5 and α = 0.1 "
             "(leftmost point is μ = 0, i.e. FedAvg; mean and 95% CI over seeds).", "experiments/fedprox-mu-sensitivity.png")

    # ------------------------------------------------------------------ EXP-03
    e3 = rr("EXP-03")
    if len(e3):
        t = mean_ci_table(e3, ["participation_rate", "algorithm"],
                          ["test_accuracy", "rounds_to_target", "participating_updates", "compute_seconds", "total_bytes", "auc_accuracy_curve"])
        t.to_csv(TAB / "exp03_participation.csv", index=False)
        md.append(md_section("EXP-03 – participation rate × algorithm (α = 0.5)", pd.DataFrame({
            "rate": t["participation_rate"], "algorithm": t["algorithm"], "n": t["n"], "test acc": fmt_ms(t, "test_accuracy"),
            "rounds→0.80": t["rounds_to_target_mean"].round(1), "client updates": t["participating_updates_mean"].round(0),
            "client compute s": t["compute_seconds_mean"].round(1), "total MB": (t["total_bytes_mean"] / 1e6).round(2)})))
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
        for i, alg in enumerate(["fedavg", "fedprox"]):
            s = t[t.algorithm == alg].sort_values("participation_rate")
            lab = "FedAvg" if alg == "fedavg" else "FedProx (μ = 1)"
            axes[0].errorbar(s["participation_rate"], s["test_accuracy_mean"], yerr=[s["test_accuracy_mean"] - s["test_accuracy_ci_low"],
                             s["test_accuracy_ci_high"] - s["test_accuracy_mean"]], color=C[i], marker="o", capsize=3, label=lab)
            axes[1].plot(s["participation_rate"], s["rounds_to_target_mean"], color=C[i], marker="o", label=lab)
            axes[2].plot(s["participation_rate"], s["compute_seconds_mean"], color=C[i], marker="o", label=lab)
        axes[0].set(xlabel="participation rate", ylabel="final test accuracy", title="Accuracy")
        axes[1].set(xlabel="participation rate", ylabel="rounds to 0.80 val acc.", title="Convergence speed")
        axes[2].set(xlabel="participation rate", ylabel="client compute (s, 40 rounds)", title="Training cost")
        axes[0].legend()
        save(fig, "exp03_participation.png",
             "Effect of the fraction of clients per round on final accuracy, rounds needed to reach 0.80 validation accuracy "
             "and total client compute (Dirichlet α = 0.5; mean over seeds, error bars = 95% CI).", "experiments/participation.png")

    # ------------------------------------------------------------------ EXP-07
    e7 = rr("EXP-07")
    if len(e7):
        t = mean_ci_table(e7, ["dropout_rate", "algorithm"], ["test_accuracy", "participating_updates", "skipped_rounds", "rounds_to_target"])
        t.to_csv(TAB / "exp07_dropout.csv", index=False)
        md.append(md_section("EXP-07 – client dropout (participation 0.5, α = 0.5)", pd.DataFrame({
            "dropout": t["dropout_rate"], "algorithm": t["algorithm"], "n": t["n"], "test acc": fmt_ms(t, "test_accuracy"),
            "updates received": t["participating_updates_mean"].round(0), "skipped rounds": t["skipped_rounds_mean"].round(2),
            "rounds→0.80": t["rounds_to_target_mean"].round(1)})))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for i, alg in enumerate(["fedavg", "fedprox"]):
            s = t[t.algorithm == alg].sort_values("dropout_rate")
            ax.errorbar(s["dropout_rate"], s["test_accuracy_mean"], yerr=[s["test_accuracy_mean"] - s["test_accuracy_ci_low"],
                        s["test_accuracy_ci_high"] - s["test_accuracy_mean"]], color=C[i], marker="o", capsize=3,
                        label="FedAvg" if alg == "fedavg" else "FedProx (μ = 1)")
        ax.set(xlabel="dropout probability of selected clients", ylabel="final global test accuracy", title="Robustness to client dropout")
        ax.legend()
        save(fig, "exp07_dropout.png",
             "Final test accuracy when each selected client independently fails to return its update with the given probability; "
             "aggregation weights are re-normalised over the survivors (mean and 95% CI over seeds).", "experiments/dropout.png")

    # ------------------------------------------------------------------ privacy
    e4 = rr("EXP-04")
    if len(e4):
        t = mean_ci_table(e4, ["variant"], ["test_accuracy", "test_macro_f1", "test_loss", "epsilon", "noise_multiplier", "volatility_last10"])
        order = ["nodp", "clip_only", "eps50", "eps20", "eps10", "eps5", "eps2"]
        t = t.set_index("variant").loc[[o for o in order if o in set(t["variant"])]].reset_index()
        t.to_csv(TAB / "exp04_privacy_budget.csv", index=False)
        md.append(md_section("EXP-04 – privacy budget sweep (logistic regression, δ = 1e-5, client-level central DP, RDP accountant)", pd.DataFrame({
            "variant": t["variant"], "n": t["n"], "σ (noise mult.)": t["noise_multiplier_mean"].round(3),
            "ε (accountant)": t["epsilon_mean"].round(2), "test acc": fmt_ms(t, "test_accuracy"),
            "macro-F1": fmt_ms(t, "test_macro_f1"), "test loss": fmt_ms(t, "test_loss")}),
            "ε is computed by the accountant for the noise actually used; 'nodp' has ε = ∞, 'clip_only' clips but adds no noise (ε = ∞)."))
        pt = paired_table(e4.assign(g="all"), ["g"], "variant", "eps10", "clip_only")
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        dp = t[t.variant.str.startswith("eps")].sort_values("epsilon_mean")
        ax.errorbar(dp["epsilon_mean"], dp["test_accuracy_mean"], yerr=[dp["test_accuracy_mean"] - dp["test_accuracy_ci_low"],
                    dp["test_accuracy_ci_high"] - dp["test_accuracy_mean"]], color=C[0], marker="o", capsize=3, label="DP-FedAvg")
        for var, col, lab in [("nodp", C[2], "no DP"), ("clip_only", C[3], "clip only (σ = 0)")]:
            r = t[t.variant == var]
            if len(r):
                ax.axhline(float(r["test_accuracy_mean"].iloc[0]), color=col, ls="--", lw=1.4, label=lab)
        ax.set_xscale("log")
        ax.set(xlabel="ε at δ = 1e-5 (client-level; smaller = stronger privacy)", ylabel="final global test accuracy", title="Privacy–utility trade-off")
        ax.legend(loc="lower right")
        save(fig, "exp04_privacy_utility.png",
             "Final test accuracy of client-level DP-FedAvg (logistic regression, 50 clients) versus the accountant's ε; dashed lines "
             "are the non-private and clip-only references (mean and 95% CI over seeds).", "experiments/privacy-utility.png")
    e5 = rr("EXP-05")
    if len(e5):
        t = mean_ci_table(e5, ["clip_norm", "noise_multiplier"], ["test_accuracy", "epsilon"])
        t.to_csv(TAB / "exp05_clip_noise.csv", index=False)
        piv = t.pivot(index="clip_norm", columns="noise_multiplier", values="test_accuracy_mean")
        eps = t.pivot(index="clip_norm", columns="noise_multiplier", values="epsilon_mean")
        md.append(md_section("EXP-05 – clipping norm × noise multiplier", pd.DataFrame({
            "clip C": t["clip_norm"], "σ": t["noise_multiplier"], "n": t["n"], "ε": t["epsilon_mean"].round(2),
            "test acc": fmt_ms(t, "test_accuracy")}), "ε depends only on σ, sampling rate and rounds (not on C)."))
        fig, ax = plt.subplots(figsize=(5.8, 3.9))
        im = ax.imshow(piv.values, cmap="Blues", aspect="auto", vmin=float(np.nanmin(piv.values)) - 0.01, vmax=float(np.nanmax(piv.values)) + 0.005)
        ax.set_xticks(range(piv.shape[1]), [f"σ={c:g}\nε={eps.iloc[0, j]:.1f}" for j, c in enumerate(piv.columns)])
        ax.set_yticks(range(piv.shape[0]), [f"C={r:g}" for r in piv.index]); ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > np.nanmean(piv.values) else INK, fontsize=9)
        ax.set_title("Test accuracy: clipping norm × noise multiplier"); fig.colorbar(im, ax=ax, fraction=0.04, label="accuracy")
        save(fig, "exp05_clip_noise_heatmap.png",
             "Mean final test accuracy for every combination of clipping norm C and noise multiplier σ (logistic regression, DP-FedAvg). "
             "Column labels give the ε that the accountant computes for each σ.", "experiments/clip-noise-heatmap.png")
    e5b = rr("EXP-05B")
    if len(e5b):
        t = mean_ci_table(e5b, ["axis_clients"], ["test_accuracy", "epsilon", "noise_multiplier"])
        t.to_csv(TAB / "exp05b_num_clients.csv", index=False)
        md.append(md_section("EXP-05B – number of clients at fixed ε", pd.DataFrame({
            "clients": t["axis_clients"], "n": t["n"], "σ": t["noise_multiplier_mean"].round(3), "ε": t["epsilon_mean"].round(2),
            "test acc": fmt_ms(t, "test_accuracy")})))
    e8 = rr("EXP-08")
    if len(e8):
        t = mean_ci_table(e8, ["axis_alpha", "algorithm"], ["test_accuracy", "epsilon", "noise_multiplier"])
        t.to_csv(TAB / "exp08_combined.csv", index=False)
        pt = paired_table(e8, ["axis_alpha"], "algorithm", "fedprox", "fedavg")
        pt.to_csv(TAB / "exp08_paired.csv", index=False)
        md.append(md_section("EXP-08 – heterogeneity + DP (ε = 10), FedAvg vs FedProx", pd.DataFrame({
            "partition": t["axis_alpha"], "algorithm": t["algorithm"], "n": t["n"], "ε": t["epsilon_mean"].round(2),
            "test acc": fmt_ms(t, "test_accuracy")})))
        if len(pt):
            md.append(md_section("EXP-08 – paired FedProx − FedAvg", pt.round(4)))
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        parts = [p for p in ["iid", "alpha0.5", "alpha0.1"] if p in set(t["axis_alpha"])]
        w = 0.36
        for i, alg in enumerate(["fedavg", "fedprox"]):
            s = t[t.algorithm == alg].set_index("axis_alpha").loc[parts]
            ax.bar(np.arange(len(parts)) + (i - 0.5) * w, s["test_accuracy_mean"], width=w - 0.03, color=C[i],
                   yerr=[s["test_accuracy_mean"] - s["test_accuracy_ci_low"], s["test_accuracy_ci_high"] - s["test_accuracy_mean"]],
                   error_kw={"ecolor": INK2, "capsize": 3}, label="FedAvg" if alg == "fedavg" else "FedProx (μ = 1)")
        ax.set_xticks(range(len(parts)), parts)
        lo = float((t["test_accuracy_ci_low"]).min())
        ax.set(ylim=(max(0.5, lo - 0.04), None), ylabel="final global test accuracy", title="Non-IID + DP (ε = 10)")
        ax.legend(loc="lower left")
        save(fig, "exp08_heterogeneity_dp.png",
             "Final test accuracy under client-level DP (target ε = 10) for IID and Dirichlet partitions, FedAvg versus FedProx "
             "(mean and 95% CI over seeds).", "experiments/heterogeneity-dp.png")

    # ------------------------------------------------------------------ EXP-06 communication
    e6 = rr("EXP-06")
    if len(e6):
        t = mean_ci_table(e6, ["variant"], ["test_accuracy", "total_bytes", "total_uplink_bytes", "mean_uplink_bytes_per_update",
                                            "uplink_compression_ratio", "compute_seconds", "auc_accuracy_curve"])
        order = ["float32", "float16", "int8", "topk10", "topk10_ef", "topk1", "topk1_ef", "int8_down16"]
        t = t.set_index("variant").loc[[o for o in order if o in set(t["variant"])]].reset_index()
        t.to_csv(TAB / "exp06_compression.csv", index=False)
        base_acc = e6[e6.variant == "float32"].set_index("seed")["test_accuracy"]
        rows = []
        for v in t["variant"]:
            o = e6[e6.variant == v].set_index("seed")["test_accuracy"]
            sd = sorted(set(o.index) & set(base_acc.index))
            from federated_intelligence.evaluation.statistical_tests import compare_paired
            r = compare_paired(o.loc[sd], base_acc.loc[sd]) if v != "float32" and len(sd) > 1 else {}
            rows.append({"variant": v, "mean_diff_vs_float32": r.get("mean_diff", 0.0),
                         "ci_low": r.get("diff_ci95", (np.nan, np.nan))[0] if r else np.nan,
                         "ci_high": r.get("diff_ci95", (np.nan, np.nan))[1] if r else np.nan, "p": r.get("p_value", np.nan)})
        pp = pd.DataFrame(rows)
        from federated_intelligence.evaluation.analysis import holm_adjust
        pp["p_holm"] = holm_adjust(pp["p"].tolist()); pp.to_csv(TAB / "exp06_paired.csv", index=False)
        md.append(md_section("EXP-06 – communication compression (measured serialized payloads)", pd.DataFrame({
            "variant": t["variant"], "n": t["n"], "bytes / update": t["mean_uplink_bytes_per_update_mean"].round(0),
            "uplink ratio": t["uplink_compression_ratio_mean"].round(2), "total MB (up+down)": (t["total_bytes_mean"] / 1e6).round(2),
            "test acc": fmt_ms(t, "test_accuracy"), "curve AUC": t["auc_accuracy_curve_mean"].round(4)}),
            "Ratios are computed from the byte length of real serialized payloads (header included)."))
        md.append(md_section("EXP-06 – paired accuracy difference to float32", pp.round(4)))
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        marks = "osD^v<>p"
        for i, r in t.iterrows():
            ax.errorbar(r["total_bytes_mean"] / 1e6, r["test_accuracy_mean"],
                        yerr=[[r["test_accuracy_mean"] - r["test_accuracy_ci_low"]], [r["test_accuracy_ci_high"] - r["test_accuracy_mean"]]],
                        marker=marks[i % len(marks)], ms=7, color=C[i % len(C)], capsize=3, ls="none")
            ax.annotate(r["variant"], (r["total_bytes_mean"] / 1e6, r["test_accuracy_mean"]), textcoords="offset points",
                        xytext=(6, 5 if i % 2 == 0 else -13), fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set(xlabel="total traffic over 40 rounds, uplink + downlink (MB, log scale)", ylabel="final global test accuracy",
               title="Accuracy vs communication volume")
        save(fig, "exp06_compression_tradeoff.png",
             "Final test accuracy against total measured traffic (uplink + downlink) for each compression scheme; "
             "points further left use fewer bytes (mean and 95% CI over seeds, MLP, α = 0.5).", "communication/compression-comparison.png")
    e9 = rr("EXP-09")
    if len(e9):
        t = mean_ci_table(e9, ["axis_deadline", "axis_codec"], ["test_accuracy", "participating_updates", "skipped_rounds",
                                                               "simulated_seconds", "wall_seconds"])
        t.to_csv(TAB / "exp09_network.csv", index=False)
        md.append(md_section("EXP-09 – simulated slow network with round deadline", pd.DataFrame({
            "deadline": t["axis_deadline"], "codec": t["axis_codec"], "n": t["n"], "test acc": fmt_ms(t, "test_accuracy"),
            "updates received": t["participating_updates_mean"].round(0), "skipped rounds": t["skipped_rounds_mean"].round(2),
            "simulated time (s)": t["simulated_seconds_mean"].round(1), "wall-clock (s)": t["wall_seconds_mean"].round(1)}),
            "Simulated time is the network model's clock (latency + size/bandwidth + compute); wall-clock is real elapsed time. They are different quantities."))
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
        order = ["noddl", "ddl2.0", "ddl1.2", "ddl0.8"]
        for i, codec in enumerate(["float32", "int8"]):
            s = t[t.axis_codec == codec].set_index("axis_deadline").loc[[o for o in order if o in set(t["axis_deadline"])]]
            xs = np.arange(len(s))
            axes[0].plot(xs, s["participating_updates_mean"], marker="o", color=C[i], label=codec)
            axes[1].plot(xs, s["test_accuracy_mean"], marker="o", color=C[i], label=codec)
            axes[0].set_xticks(xs, s.index); axes[1].set_xticks(xs, s.index)
        axes[0].set(ylabel="client updates accepted (40 rounds)", title="Stragglers dropped by the deadline", xlabel="round deadline")
        axes[1].set(ylabel="final test accuracy", title="Accuracy under deadlines", xlabel="round deadline")
        axes[0].legend(title="uplink codec")
        save(fig, "exp09_network_deadline.png",
             "Simulated 0.5 Mbit/s heterogeneous network with 5% communication failures: number of client updates accepted and final "
             "accuracy as the round deadline tightens, for float32 and int8 uplinks.", "communication/network-deadline.png")

    # ------------------------------------------------------------------ write
    (TAB / "results_summary.md").write_text("\n".join(md))
    (FIG / "captions.json").write_text(json.dumps(CAPTIONS, indent=2))
    for name, dest in COPIES.items():
        d = DOCS / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIG / name, d)
    # curated dataset figures
    src = ROOT / "reports" / "figures" / "data-profiling"
    for s, d in [("partition_dirichlet_alpha_0_1_label-heatmap.png", "datasets/heterogeneity-heatmap.png"),
                 ("scenario_combined_non_iid_sample-imbalance.png", "datasets/client-distribution.png")]:
        if (src / s).exists():
            (DOCS / d).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / s, DOCS / d)
    print(f"{len(runs)} runs, {len(CAPTIONS)} figures, tables -> {TAB}")


if __name__ == "__main__":
    main()
