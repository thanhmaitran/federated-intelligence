"""Import persisted run files (reports/experiment-results/**.json) into the database."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sqlalchemy.orm import Session

from federated_intelligence.api.db import Experiment, RoundMetric
from federated_intelligence.evaluation.analysis import iter_result_files


def run_id(data: dict) -> str:
    return f"{data.get('exp_id') or 'adhoc'}/{data.get('variant') or data['name']}__seed{data['seed']}"


def store_run(session: Session, data: dict, source: str = "imported", exp_id: str | None = None) -> Experiment:
    """Insert (or replace) one run. ``data`` is the JSON written by ``run_experiment``."""
    eid = exp_id or run_id(data)
    old = session.get(Experiment, eid)
    if old is not None:
        session.delete(old)
        session.flush()
    exp = Experiment(
        id=eid, exp_group=data.get("exp_id", ""), variant=data.get("variant") or data["name"], seed=data["seed"],
        status="finished", source=source, config={**data["config"], "axes": data.get("axes", {})},
        summary=data["summary"], per_client=data.get("per_client_final", {}),
        finished_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
    )
    for h in data["history"]:
        exp.rounds.append(RoundMetric(round=h["round"], metrics=h))
    session.add(exp)
    return exp


def import_results(session: Session, results_dir: str | Path, prefix: str | None = "EXP-") -> int:
    n = 0
    for f in iter_result_files(results_dir, prefix):
        data = json.loads(f.read_text())
        if "exp_id" not in data:
            continue
        store_run(session, data)
        n += 1
        if n % 25 == 0:
            session.commit()
    session.commit()
    return n
