"""Figures for data profiling. Every figure is saved together with a caption."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from federated_intelligence.data.federated_dataset import FederatedDataset  # noqa: E402
from federated_intelligence.data.profiling.heterogeneity import (  # noqa: E402
    jsd_matrix,
    label_distribution_matrix,
)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def _save(fig, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    captions_path = path.parent / "captions.json"
    captions = json.loads(captions_path.read_text()) if captions_path.exists() else {}
    captions[path.name] = caption
    captions_path.write_text(json.dumps(captions, indent=2))


def plot_label_heatmap(ds: FederatedDataset, path: Path, title: str = "") -> None:
    dist = label_distribution_matrix(ds)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(dist.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("client index")
    ax.set_ylabel("class")
    ax.set_yticks(range(dist.shape[1]))
    ax.set_title(title or "Per-client class proportions (training data)")
    fig.colorbar(im, ax=ax, label="proportion")
    _save(
        fig,
        path,
        "Heatmap of class proportions per client. Uniform colour across columns indicates IID "
        "labels; strongly varying columns indicate label skew.",
    )


def plot_feature_distribution(
    ds: FederatedDataset, path: Path, features: tuple[int, ...] = (0, 1, 2, 3)
) -> None:
    ids = ds.client_ids
    fig, axes = plt.subplots(1, len(features), figsize=(2.6 * len(features), 3.2), sharey=False)
    for ax, f in zip(np.atleast_1d(axes), features, strict=True):
        data = [ds.clients[c].x_train[:, f] for c in ids]
        ax.boxplot(data, showfliers=False, widths=0.6)
        ax.set_title(f"feature {f}")
        ax.set_xticks([])
        ax.set_xlabel("clients")
    _save(
        fig,
        path,
        "Distribution of four features for every client (box = interquartile range). Shifts of "
        "the boxes between clients show feature (covariate) skew.",
    )


def plot_sample_imbalance(ds: FederatedDataset, path: Path) -> None:
    n = np.array([ds.clients[c].num_train for c in ds.client_ids])
    order = np.argsort(-n)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(n)), n[order], color="#3b6ea8")
    ax.axhline(n.mean(), color="#c0392b", ls="--", lw=1, label=f"mean = {n.mean():.0f}")
    ax.set_xlabel("clients (sorted by size)")
    ax.set_ylabel("training samples")
    ax.set_title(f"Samples per client (CV = {n.std() / n.mean():.2f})")
    ax.legend(frameon=False)
    _save(
        fig,
        path,
        "Training-set size of every client, sorted. A large spread (high coefficient of variation) "
        "indicates quantity skew.",
    )


def plot_client_divergence(ds: FederatedDataset, path: Path) -> None:
    jsd = jsd_matrix(label_distribution_matrix(ds))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(jsd, cmap="magma", vmin=0, vmax=max(0.05, jsd.max()))
    ax.set_xlabel("client")
    ax.set_ylabel("client")
    ax.set_title("Pairwise label JSD (base 2)")
    fig.colorbar(im, ax=ax, label="JSD")
    _save(
        fig,
        path,
        "Pairwise Jensen-Shannon divergence between client label distributions (0 = identical, "
        "1 = disjoint). Bright blocks mark client groups with very different label mixes.",
    )


def profile_figures(ds: FederatedDataset, out_dir: Path, prefix: str) -> list[Path]:
    out_dir = Path(out_dir)
    paths: dict[str, Callable[[FederatedDataset, Path], object]] = {
        "label-heatmap": plot_label_heatmap,
        "feature-distribution": plot_feature_distribution,
        "sample-imbalance": plot_sample_imbalance,
        "client-divergence": plot_client_divergence,
    }
    written = []
    for name, fn in paths.items():
        p = out_dir / f"{prefix}_{name}.png"
        fn(ds, p)
        written.append(p)
    return written
