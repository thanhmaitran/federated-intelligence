import numpy as np
import pytest
import torch

from federated_intelligence.evaluation.convergence import (
    area_under_curve,
    curve_volatility,
    rounds_to_target,
)
from federated_intelligence.evaluation.metrics import classification_metrics, perplexity
from federated_intelligence.evaluation.statistical_tests import (
    bootstrap_ci,
    cohens_d,
    compare_independent,
    compare_paired,
    summarize,
)
from federated_intelligence.models import build_model, count_parameters


def test_model_shapes():
    assert build_model("logreg", num_features=6)(torch.zeros(4, 6)).shape == (4, 2)
    assert build_model("mlp", num_features=6, hidden=[8, 4])(torch.zeros(4, 6)).shape == (4, 2)
    # FEMNIST CNN and Shakespeare GRU: architectures validated on random tensors.
    assert build_model("cnn")(torch.zeros(3, 1, 28, 28)).shape == (3, 62)
    assert build_model("gru", vocab_size=80)(torch.zeros(3, 80, dtype=torch.long)).shape == (3, 80)
    assert count_parameters(build_model("cnn")) > 100_000
    with pytest.raises(ValueError):
        build_model("transformer")


def test_cnn_and_gru_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    for name, x, y in [
        ("cnn", torch.randn(8, 1, 28, 28), torch.randint(0, 62, (8,))),
        ("gru", torch.randint(0, 80, (8, 20)), torch.randint(0, 80, (8,))),
    ]:
        m = build_model(name)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        first = None
        for _ in range(60):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(x), y)
            first = first if first is not None else float(loss)
            loss.backward()
            opt.step()
        assert float(loss) < 0.5 * first


def test_binary_metrics_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    logits = np.array([[3, 0], [2, 0], [0, 2], [0, 3]], dtype=float)
    m = classification_metrics(logits, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0 and m["loss"] < 0.2


def test_metrics_multiclass_has_no_auc():
    m = classification_metrics(np.eye(5) * 5, np.arange(5))
    assert "roc_auc" not in m and m["accuracy"] == 1.0


def test_perplexity():
    assert perplexity(0.0) == 1.0
    assert perplexity(np.log(10)) == pytest.approx(10.0)


def test_summarize_known_values():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s.mean == 3.0 and s.sd == pytest.approx(np.sqrt(2.5))
    assert s.ci_low < 3.0 < s.ci_high
    assert s.ci_high - 3.0 == pytest.approx(2.776445 * np.sqrt(2.5) / np.sqrt(5), rel=1e-4)
    single = summarize([2.0])
    assert single.sd == 0.0 and single.ci_low == single.ci_high == 2.0


def test_cohens_d_sign_and_magnitude():
    a = [10.0, 11.0, 12.0, 13.0]
    b = [0.0, 1.0, 2.0, 3.0]
    assert cohens_d(a, b) > 5
    assert cohens_d(b, a) < -5


def test_paired_and_independent_tests_detect_real_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(0.8, 0.01, 10)
    better = base + 0.05 + rng.normal(0, 0.005, 10)
    res = compare_paired(better, base)
    assert res["significant"] and res["mean_diff"] > 0.04 and res["cohens_dz"] > 1
    assert compare_independent(better, base)["significant"]
    null = compare_paired(base, base + rng.normal(0, 0.01, 10))
    assert null["p_value"] > 0.01


def test_paired_small_n_flagged_exploratory():
    assert "exploratory" in compare_paired([0.8, 0.82, 0.81], [0.7, 0.72, 0.69])["note"]


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(1).normal(5, 1, 30)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_convergence_helpers():
    r = [0, 1, 2, 3, 4]
    v = [0.5, 0.6, 0.75, 0.8, 0.8]
    assert rounds_to_target(r, v, 0.75) == 2
    assert rounds_to_target(r, v, 0.95) is None
    assert 0.5 < area_under_curve(r, v) < 0.8
    assert curve_volatility([0.8, 0.8, 0.8]) == pytest.approx(0.0, abs=1e-9)
