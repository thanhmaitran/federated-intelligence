"""Statistics across independent seeds: mean, SD, confidence intervals, effect sizes, tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Summary:
    n: int
    mean: float
    sd: float
    ci_low: float
    ci_high: float

    def fmt(self, digits: int = 4) -> str:
        return f"{self.mean:.{digits}f} ± {self.sd:.{digits}f}"


def summarize(values: Sequence[float], confidence: float = 0.95) -> Summary:
    """Mean, sample SD (n-1) and two-sided Student-t confidence interval."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n))
    else:
        half = 0.0
    return Summary(n, mean, sd, mean - half, mean + half)


def cohens_d(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cohen's d with pooled SD (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def paired_effect_size(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cohen's d_z for paired samples (same seeds in both arms)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else float("nan")


def compare_paired(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Paired comparison (same seeds): t-test + Wilcoxon + effect size + CI of the difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diff = a - b
    res: dict = {
        "n": len(a),
        "mean_diff": float(diff.mean()),
        "diff_ci95": (
            summarize(diff).ci_low,
            summarize(diff).ci_high,
        ),
        "cohens_dz": paired_effect_size(a, b),
    }
    if len(a) >= 2 and diff.std(ddof=1) > 0:
        t = stats.ttest_rel(a, b)
        res.update(t_stat=float(t.statistic), p_value=float(t.pvalue))
        res["significant"] = bool(t.pvalue < alpha)
        if len(a) >= 5:
            try:
                res["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
            except ValueError:
                res["wilcoxon_p"] = float("nan")
    else:
        res.update(t_stat=float("nan"), p_value=float("nan"), significant=False)
    res["note"] = (
        "n < 5 seeds: tests have very low power; treat as exploratory" if len(a) < 5 else ""
    )
    return res


def compare_independent(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Welch's t-test for independent samples, with Cohen's d."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
        "cohens_d": cohens_d(a, b),
        "significant": bool(t.pvalue < alpha),
    }


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 5000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, float)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return float(lo), float(hi)
