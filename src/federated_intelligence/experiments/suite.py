"""Experiment suites: YAML-defined grids of runs, expanded over variants and seeds.

A suite file looks like::

    experiments:
      EXP-02:
        description: ...
        research_question: RQ2
        overrides: {...}            # applied to every variant of this experiment
        grid:                       # cartesian product of named axes
          alpha:  [{name: a0.5, overrides: {dataset: {alpha: 0.5}}}, ...]
          algo:   [{name: fedavg, overrides: {...}}, {name: fedprox, overrides: {...}}]
        variants:                   # optional explicit variants (added after the grid)
          - {name: ..., overrides: {...}}

``_defaults.yaml`` (next to the suite) provides the base configuration shared by all suites.
Every run is independent, deterministic given its seed and written to
``<results_dir>/<EXP-ID>/<variant>__seed<k>.json``; finished runs are skipped on re-execution so
a suite can be resumed after an interruption.
"""

from __future__ import annotations

import copy
import itertools
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from federated_intelligence.experiments.configuration import ExperimentConfig, _deep_update


@dataclass
class RunSpec:
    exp_id: str
    variant: str
    seed: int
    config: dict[str, Any]
    out_dir: Path
    tracker: str = "none"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def result_path(self) -> Path:
        return self.out_dir / f"{self.variant}__seed{self.seed}.json"


def load_suite(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suite = yaml.safe_load(path.read_text()) or {}
    defaults_path = path.parent / "_defaults.yaml"
    defaults = yaml.safe_load(defaults_path.read_text()) if defaults_path.exists() else {}
    suite["defaults"] = _deep_update(copy.deepcopy(defaults), suite.get("defaults", {}))
    suite["source"] = str(path)
    return suite


def _variants(exp: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return (variant name, overrides, axis labels) triples."""
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    grid = exp.get("grid")
    if grid:
        axes = list(grid.items())
        for combo in itertools.product(*[opts for _, opts in axes]):
            name = "__".join(o["name"] for o in combo)
            ov: dict[str, Any] = {}
            for o in combo:
                _deep_update(ov, copy.deepcopy(o.get("overrides", {})))
            labels = {axis: o["name"] for (axis, _), o in zip(axes, combo, strict=True)}
            out.append((name, ov, labels))
    for v in exp.get("variants", []) or []:
        out.append((v["name"], copy.deepcopy(v.get("overrides", {})), {"variant": v["name"]}))
    if not out:
        out.append(("default", {}, {}))
    return out


def expand_suite(
    suite: dict[str, Any],
    results_dir: str | Path,
    exp_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    smoke: bool = False,
    tracker: str = "none",
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for exp_id, exp in suite["experiments"].items():
        if exp_ids and exp_id not in exp_ids:
            continue
        run_seeds = seeds or exp.get("seeds") or suite.get("seeds") or [1, 2, 3, 4, 5]
        for name, ov, labels in _variants(exp):
            for seed in run_seeds:
                cfg = copy.deepcopy(suite["defaults"])
                _deep_update(cfg, copy.deepcopy(exp.get("overrides", {})))
                _deep_update(cfg, ov)
                if smoke:
                    _deep_update(cfg, {"federated": {"num_rounds": 3}})
                    cfg.setdefault("dataset", {})["total_samples"] = min(
                        4000, cfg.get("dataset", {}).get("total_samples", 4000)
                    )
                cfg.setdefault("experiment", {}).update(
                    {"name": name, "seed": seed, "description": exp.get("description", "")}
                )
                ExperimentConfig.model_validate(cfg)  # fail early on invalid configs
                out = Path(results_dir) / ("smoke" if smoke else "") / exp_id
                specs.append(
                    RunSpec(exp_id, name, seed, cfg, out, tracker,
                            {"axes": labels, "research_question": exp.get("research_question", "")})
                )
    return specs


def _execute(spec: RunSpec) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from federated_intelligence.experiments.runner import run_experiment
    from federated_intelligence.experiments.tracking import build_tracker

    cfg = ExperimentConfig.model_validate(spec.config)
    tracker = build_tracker(spec.tracker, spec.out_dir, f"{spec.exp_id}/{spec.variant}")
    t0 = time.perf_counter()
    res = run_experiment(cfg, out_dir=spec.out_dir, tracker=tracker)
    # Annotate the persisted file with suite metadata (axes etc.) for downstream analysis.
    path = Path(res["result_path"])
    data = json.loads(path.read_text())
    data["exp_id"], data["variant"], data["axes"] = spec.exp_id, spec.variant, spec.meta.get("axes", {})
    data["research_question"] = spec.meta.get("research_question", "")
    path.write_text(json.dumps(data, default=float, allow_nan=False))
    return {"path": str(path), "seconds": time.perf_counter() - t0,
            "acc": res["summary"]["final"]["test_accuracy"]}


def run_suite(
    specs: list[RunSpec], workers: int = 2, skip_existing: bool = True, progress: bool = True
) -> list[Path]:
    todo = [s for s in specs if not (skip_existing and s.result_path.exists())]
    done = [s.result_path for s in specs if s not in todo]
    if progress:
        print(f"[suite] {len(specs)} runs: {len(done)} cached, {len(todo)} to execute", flush=True)
    if not todo:
        return done
    t0 = time.perf_counter()
    if workers <= 1:
        for i, s in enumerate(todo, 1):
            r = _execute(s)
            done.append(Path(r["path"]))
            if progress:
                print(f"[{i}/{len(todo)}] {s.exp_id}/{s.variant} seed={s.seed} acc={r['acc']:.4f} ({r['seconds']:.1f}s)", flush=True)
        return done
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(_execute, s): s for s in todo}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            r = f.result()
            done.append(Path(r["path"]))
            if progress:
                el = time.perf_counter() - t0
                print(f"[{i}/{len(todo)}] {s.exp_id}/{s.variant} seed={s.seed} acc={r['acc']:.4f} "
                      f"({r['seconds']:.1f}s, elapsed {el/60:.1f} min)", flush=True)
    return done
