"""REST routes: health, datasets, clients, experiments, analysis."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from federated_intelligence import __version__
from federated_intelligence.api import schemas as S
from federated_intelligence.api.db import Experiment, RoundMetric, utcnow
from federated_intelligence.api.importer import store_run
from federated_intelligence.evaluation.statistical_tests import summarize
from federated_intelligence.experiments.configuration import ExperimentConfig

router = APIRouter()


def get_db(request: Request):
    s = request.app.state.session_factory()
    try:
        yield s
    finally:
        s.close()


DB = Depends(get_db)


def _dig(d: dict, path: str, default=None):
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _summary_out(e: Experiment) -> S.ExperimentSummary:
    fin = (e.summary or {}).get("final", {})
    eps = _dig(e.summary or {}, "privacy.epsilon")
    return S.ExperimentSummary(
        id=e.id, exp_group=e.exp_group, variant=e.variant, seed=e.seed, status=e.status, source=e.source,
        algorithm=_dig(e.config or {}, "federated.algorithm"), test_accuracy=fin.get("test_accuracy"),
        total_bytes=(e.summary or {}).get("total_bytes"), epsilon=eps if eps is not None and np.isfinite(eps) else None,
        created_at=e.created_at, finished_at=e.finished_at,
    )


# ------------------------------------------------------------------ health
@router.get("/health", response_model=S.Health)
def health(db: Session = DB) -> S.Health:
    db.execute(select(1))
    return S.Health(status="ok", version=__version__, database=db.get_bind().dialect.name)


# ------------------------------------------------------------------ datasets & clients
def _meta_dir(request: Request) -> Path:
    return Path(request.app.state.data_dir) / "metadata"


@router.get("/datasets", response_model=list[S.DatasetInfo])
def datasets(request: Request) -> list[S.DatasetInfo]:
    meta = _meta_dir(request)
    out: list[S.DatasetInfo] = []
    for sc in ("iid", "label_skew", "feature_skew", "quantity_skew", "combined_non_iid"):
        p = meta / f"synthetic_{sc}.json"
        info = json.loads(p.read_text()) if p.exists() else {}
        out.append(S.DatasetInfo(
            name=f"synthetic/{sc}", kind="synthetic", description=f"Synthetic federated tabular data, scenario '{sc}'.",
            status="generated" if info else "not_generated", num_clients=info.get("num_clients"),
            metadata={k: v for k, v in info.items() if k in ("seed", "num_features", "num_classes", "generator_version")}))
    for name, desc in (("femnist", "LEAF FEMNIST: 62-class handwritten characters, natural partition by writer."),
                       ("shakespeare", "LEAF Shakespeare: next-character prediction, natural partition by role.")):
        p = meta / f"{name}.json"
        info = json.loads(p.read_text()) if p.exists() else {}
        out.append(S.DatasetInfo(
            name=name, kind="natural", description=desc, status="available" if info else "not_downloaded",
            num_clients=info.get("num_clients_in_source"),
            metadata={k: info[k] for k in ("source", "license", "num_rows", "accessed_utc") if k in info}))
    return out


@router.get("/partitions", response_model=list[S.PartitionInfo])
def partitions(request: Request) -> list[S.PartitionInfo]:
    out = []
    for p in sorted(_meta_dir(request).glob("partition_*.json")):
        m = json.loads(p.read_text())
        out.append(S.PartitionInfo(partition_id=m["partition_id"], dataset=m["dataset"], method=m["partition_method"],
                                   alpha=m.get("alpha"), num_clients=m["num_clients"], total_samples=m.get("total_samples", 0)))
    return out


@router.get("/clients", response_model=list[S.ClientInfo])
def clients(request: Request, partition_id: str | None = None, experiment_id: str | None = None,
            db: Session = DB) -> list[S.ClientInfo]:
    """Client metadata of a partition manifest, or per-client results of an experiment."""
    if experiment_id:
        e = db.get(Experiment, experiment_id)
        if e is None:
            raise HTTPException(404, "experiment not found")
        return [S.ClientInfo(client_id=cid, num_samples=int(m.get("num_samples", 0)) or None,
                             metrics={k: float(v) for k, v in m.items() if k != "num_samples" and v is not None})
                for cid, m in sorted((e.per_client or {}).items())]
    pid = partition_id or "dirichlet_alpha_0_5_seed_42"
    path = _meta_dir(request) / f"partition_{pid}.json"
    if not path.exists():
        raise HTTPException(404, f"partition '{pid}' not found")
    m = json.loads(path.read_text())
    return [S.ClientInfo(client_id=cid, num_samples=c["num_samples"], train_samples=c["train_samples"],
                         validation_samples=c["validation_samples"], test_samples=c["test_samples"])
            for cid, c in sorted(m["clients"].items())]


# ------------------------------------------------------------------ experiments
def _run_in_background(app, exp_id: str, cfg: ExperimentConfig, group: str) -> None:
    from federated_intelligence.experiments.runner import run_experiment

    factory = app.state.session_factory
    with factory() as s:
        e = s.get(Experiment, exp_id)
        e.status = "running"
        s.commit()
    try:
        result = run_experiment(cfg)
        result["exp_id"], result["variant"] = group, cfg.experiment.name
        with factory() as s:
            s.delete(s.get(Experiment, exp_id))
            s.flush()
            store_run(s, result, source="api", exp_id=exp_id)
            s.commit()
    except Exception as exc:
        with factory() as s:
            e = s.get(Experiment, exp_id)
            e.status, e.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
            e.finished_at = utcnow()
            s.commit()


@router.post("/experiments", response_model=S.ExperimentDetail, status_code=202)
def create_experiment(body: S.ExperimentCreate, request: Request, db: Session = DB) -> S.ExperimentDetail:
    try:
        cfg = ExperimentConfig.model_validate(body.config)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    cfg.execution.backend = "local"  # the API never starts a Ray cluster
    exp_id = f"{body.group}/{cfg.experiment.name}__seed{cfg.experiment.seed}__{utcnow():%H%M%S%f}"
    e = Experiment(id=exp_id, exp_group=body.group, variant=cfg.experiment.name, seed=cfg.experiment.seed,
                   status="queued", source="api", config=cfg.model_dump(mode="json"))
    db.add(e)
    db.commit()
    if not request.app.state.run_inline:
        threading.Thread(target=_run_in_background, args=(request.app, exp_id, cfg, body.group), daemon=True).start()
    else:
        _run_in_background(request.app, exp_id, cfg, body.group)
    db.refresh(e)
    return S.ExperimentDetail(**_summary_out(e).model_dump(), config=e.config, summary=e.summary or {}, error=e.error)


@router.get("/experiments", response_model=list[S.ExperimentSummary])
def list_experiments(group: str | None = None, status: str | None = None, algorithm: str | None = None,
                     limit: int = Query(200, le=2000), offset: int = 0, db: Session = DB) -> list[S.ExperimentSummary]:
    q = select(Experiment).order_by(Experiment.exp_group, Experiment.variant, Experiment.seed)
    if group:
        q = q.where(Experiment.exp_group == group)
    if status:
        q = q.where(Experiment.status == status)
    rows: list[Experiment] = list(db.scalars(q.offset(offset).limit(limit)).all())
    out = [_summary_out(e) for e in rows]
    return [o for o in out if algorithm is None or o.algorithm == algorithm]


@router.get("/experiments/{exp_id:path}/metrics")
def experiment_metrics(exp_id: str, db: Session = DB) -> dict[str, Any]:
    """Round-wise series of every logged scalar metric: ``{metric: [{round, value}, ...]}``."""
    e = db.get(Experiment, exp_id)
    if e is None:
        raise HTTPException(404, "experiment not found")
    rows: list[RoundMetric] = list(
        db.scalars(select(RoundMetric).where(RoundMetric.experiment_id == exp_id).order_by(RoundMetric.round))
    )
    series: dict[str, list[dict[str, float]]] = {}
    for r in rows:
        for k, v in r.metrics.items():
            if k != "round" and isinstance(v, (int, float)) and not isinstance(v, bool):
                series.setdefault(k, []).append({"round": r.round, "value": v})
    return {"experiment_id": exp_id, "status": e.status, "metrics": series}


@router.get("/experiments/{exp_id:path}/rounds", response_model=list[S.RoundOut])
def experiment_rounds(exp_id: str, db: Session = DB) -> list[S.RoundOut]:
    if db.get(Experiment, exp_id) is None:
        raise HTTPException(404, "experiment not found")
    rows: list[RoundMetric] = list(
        db.scalars(select(RoundMetric).where(RoundMetric.experiment_id == exp_id).order_by(RoundMetric.round))
    )
    return [S.RoundOut(round=r.round, metrics=r.metrics) for r in rows]


@router.get("/experiments/{exp_id:path}", response_model=S.ExperimentDetail)
def experiment_detail(exp_id: str, db: Session = DB) -> S.ExperimentDetail:
    e = db.get(Experiment, exp_id)
    if e is None:
        raise HTTPException(404, "experiment not found")
    return S.ExperimentDetail(**_summary_out(e).model_dump(), config=e.config, summary=e.summary or {}, error=e.error)


# ------------------------------------------------------------------ analysis
@router.get("/overview", response_model=S.Overview)
def overview(request: Request, db: Session = DB) -> S.Overview:
    rows: list[Experiment] = list(db.scalars(select(Experiment)).all())
    accs = [r.summary.get("final", {}).get("test_accuracy") for r in rows if r.summary]
    accs = [a for a in accs if a is not None]
    by_status: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_group[r.exp_group] = by_group.get(r.exp_group, 0) + 1
    total = sum(int(r.summary.get("total_bytes", 0)) for r in rows if r.summary)
    return S.Overview(num_experiments=len(rows), by_status=by_status, by_group=by_group,
                      best_test_accuracy=max(accs) if accs else None, total_bytes_transferred=total,
                      datasets=len(datasets(request)))


_METRIC_PATHS = {
    "test_accuracy": "summary.final.test_accuracy", "test_macro_f1": "summary.final.test_macro_f1",
    "test_loss": "summary.final.test_loss", "total_bytes": "summary.total_bytes",
    "epsilon": "summary.privacy.epsilon", "rounds_to_target": "summary.rounds_to_target_val_accuracy",
    "client_acc_min": "summary.final.client_acc_min", "skipped_rounds": "summary.skipped_rounds",
}


@router.get("/analysis/compare", response_model=list[S.ComparisonRow])
def compare(group: str, metric: str = "test_accuracy",
            by: str = Query("variant", description="comma separated: variant | any config path such as federated.algorithm"),
            db: Session = DB) -> list[S.ComparisonRow]:
    """Mean, SD and 95% t-CI of ``metric`` over seeds, grouped by ``by`` (finished runs of ``group``)."""
    if metric not in _METRIC_PATHS:
        raise HTTPException(422, f"metric must be one of {sorted(_METRIC_PATHS)}")
    keys = [k.strip() for k in by.split(",") if k.strip()]
    groups: dict[tuple, list[float]] = {}
    finished: list[Experiment] = list(
        db.scalars(select(Experiment).where(Experiment.exp_group == group, Experiment.status == "finished"))
    )
    for e in finished:
        full = {"summary": e.summary, "config": e.config}
        v = _dig(full, _METRIC_PATHS[metric])
        if v is None or not np.isfinite(v):
            continue
        key = tuple(e.variant if k == "variant" else _dig(e.config, k) for k in keys)
        groups.setdefault(key, []).append(float(v))
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        s = summarize(vals)
        out.append(S.ComparisonRow(key=dict(zip(keys, key, strict=True)), n=s.n, mean=s.mean, sd=s.sd,
                                   ci_low=s.ci_low, ci_high=s.ci_high))
    return out


@router.get("/analysis/groups")
def groups(db: Session = DB) -> list[dict[str, Any]]:
    rows = db.execute(select(Experiment.exp_group, func.count(Experiment.id)).group_by(Experiment.exp_group)
                      .order_by(Experiment.exp_group)).all()
    return [{"group": g, "runs": n} for g, n in rows]
