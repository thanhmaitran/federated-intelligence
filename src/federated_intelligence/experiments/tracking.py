"""Experiment tracking. A local JSON tracker is always available; MLflow is optional."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tracker:
    def start(self, name: str, params: dict[str, Any]) -> None: ...
    def log_round(self, row: dict[str, Any]) -> None: ...
    def finish(self, summary: dict[str, Any], artifacts: list[Path] | None = None) -> None: ...


class NullTracker(Tracker):
    pass


class JsonlTracker(Tracker):
    """Append-only JSONL log of rounds: tail it to monitor a running experiment."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start(self, name, params):
        self.path.write_text(json.dumps({"event": "start", "name": name, "params": params}) + "\n")

    def log_round(self, row):
        with self.path.open("a") as f:
            f.write(json.dumps({"event": "round", **row}, default=float) + "\n")

    def finish(self, summary, artifacts=None):
        with self.path.open("a") as f:
            f.write(json.dumps({"event": "finish", "summary": summary}, default=float) + "\n")


class MLflowTracker(Tracker):
    """Logs parameters, per-round metrics and artifacts to MLflow (optional dependency)."""

    def __init__(self, experiment: str = "federated-intelligence", tracking_uri: str | None = None):
        import mlflow

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    @staticmethod
    def _flat(d: dict, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(MLflowTracker._flat(v, key + "."))
            else:
                out[key] = v
        return out

    def start(self, name, params):
        self.mlflow.start_run(run_name=name)
        self.mlflow.log_params({k: str(v)[:250] for k, v in self._flat(params).items()})

    def log_round(self, row):
        step = int(row["round"])
        metrics = {
            k: float(v)
            for k, v in row.items()
            if isinstance(v, (int, float)) and k != "round" and v == v
        }
        self.mlflow.log_metrics(metrics, step=step)

    def finish(self, summary, artifacts=None):
        flat = {k: v for k, v in self._flat(summary).items() if isinstance(v, (int, float))}
        self.mlflow.log_metrics({f"summary.{k}": float(v) for k, v in flat.items() if v == v})
        for a in artifacts or []:
            self.mlflow.log_artifact(str(a))
        self.mlflow.end_run()


def build_tracker(kind: str, out_dir: Path | None, name: str, mlflow_uri: str | None = None) -> Tracker:
    if kind == "mlflow":
        return MLflowTracker(tracking_uri=mlflow_uri)
    if kind == "jsonl" and out_dir is not None:
        return JsonlTracker(Path(out_dir) / f"{name}.jsonl")
    return NullTracker()
