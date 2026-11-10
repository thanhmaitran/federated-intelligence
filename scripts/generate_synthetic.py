#!/usr/bin/env python
"""Generate all synthetic scenarios + Dirichlet/quantity partitions, validate and profile them.

Outputs
-------
data/synthetic/<scenario>/{clients.parquet,global_test.parquet,manifest.json}   (git-ignored data)
data/metadata/synthetic_<scenario>.json, data/metadata/partition_*.json          (committed manifests)
reports/figures/data-profiling/*.png (+ captions.json)
reports/tables/heterogeneity_by_scenario.csv, data_quality_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from federated_intelligence.data.generators import GeneratorConfig, Scenario, TabularGenerator
from federated_intelligence.data.partitioners import build_manifest
from federated_intelligence.data.pooled import build_partitioned_dataset, make_partitioner, sample_population
from federated_intelligence.data.profiling import heterogeneity_summary
from federated_intelligence.data.profiling.plots import profile_figures
from federated_intelligence.data.validators import check_dataset, check_partition

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/datasets/synthetic.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gen, het = cfg["generation"], cfg["heterogeneity"]
    seed = args.seed if args.seed is not None else gen["seed"]
    base = dict(
        seed=seed,
        num_clients=gen["num_clients"],
        num_features=gen["num_features"],
        min_samples_per_client=gen["min_samples_per_client"],
        max_samples_per_client=gen["max_samples_per_client"],
        weight_scale=gen["weight_scale"],
        global_test_size=gen["global_test_size"],
        label_beta=het["label_beta"],
        feature_shift=het["feature_shift"],
        quantity_sigma=het["quantity_sigma"],
        train_fraction=cfg["splits"]["train"],
        val_fraction=cfg["splits"]["validation"],
        test_fraction=cfg["splits"]["test"],
    )
    fig_dir = ROOT / "reports/figures/data-profiling"
    meta_dir = ROOT / "data/metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rows, quality = [], {}

    # --- native scenarios -------------------------------------------------------------
    for name in het["scenarios"]:
        gcfg = GeneratorConfig(**base, scenario=Scenario(name))
        g = TabularGenerator(gcfg)
        ds = g.generate()
        manifest = g.write(ROOT / "data/synthetic" / name, ds)
        (meta_dir / f"synthetic_{name}.json").write_text(
            json.dumps({k: v for k, v in manifest.items() if k != "client_parameters"}, indent=2)
        )
        rep = check_dataset(ds)
        quality[f"scenario:{name}"] = rep.to_dict()
        rows.append({"dataset": f"scenario:{name}", **heterogeneity_summary(ds)})
        if not args.no_figures:
            profile_figures(ds.standardized(), fig_dir, f"scenario_{name}")
        print(f"[scenario {name}] quality={'OK' if rep.passed else rep.errors}")

    # --- pooled population partitioned by Dirichlet / quantity skew ----------------------
    gcfg = GeneratorConfig(**base, scenario=Scenario.IID)
    n = het["pooled_total_samples"]
    _, y = sample_population(gcfg, n)
    plans = [("iid", {})] + [("dirichlet", {"alpha": a}) for a in het["dirichlet_alphas"]] + [
        ("quantity_skew", {"sigma": het["quantity_sigma"]})
    ]
    for method, kw in plans:
        tag = method + (f"_alpha_{kw['alpha']}".replace(".", "_") if "alpha" in kw else "")
        part = make_partitioner(method, gcfg.num_clients, seed, **kw)
        partition = part.partition(y)
        pr = check_partition(partition, n)
        quality[f"partition:{tag}"] = pr.to_dict()
        manifest = build_manifest(f"{tag}_seed_{seed}", "synthetic_federated_tabular", part, partition, include_indices=False)
        manifest["total_samples"] = n
        (meta_dir / f"partition_{tag}_seed_{seed}.json").write_text(json.dumps(manifest, indent=2))
        ds = build_partitioned_dataset(gcfg, method, n, **kw)
        rows.append({"dataset": f"partition:{tag}", **heterogeneity_summary(ds)})
        if not args.no_figures and method in ("iid", "dirichlet"):
            profile_figures(ds.standardized(), fig_dir, f"partition_{tag}")
        print(f"[partition {tag}] covers all samples: {pr.passed}")

    tables = ROOT / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "heterogeneity_by_scenario.csv", index=False)
    (tables / "data_quality_report.json").write_text(json.dumps(quality, indent=2, default=float))
    failed = [k for k, v in quality.items() if not v["passed"]]
    print("quality failures:", failed or "none")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
