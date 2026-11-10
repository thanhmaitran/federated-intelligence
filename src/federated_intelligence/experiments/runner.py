"""Experiment runner: config -> dataset -> executor -> coordinator -> JSON result."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from federated_intelligence.clients.client import FederatedClient
from federated_intelligence.communication.compression import build_compressor
from federated_intelligence.data.federated_dataset import FederatedDataset
from federated_intelligence.data.generators import GeneratorConfig, Scenario, generate_dataset
from federated_intelligence.data.pooled import build_partitioned_dataset
from federated_intelligence.experiments.configuration import ExperimentConfig
from federated_intelligence.experiments.reproducibility import environment_info
from federated_intelligence.experiments.tracking import Tracker, build_tracker
from federated_intelligence.models.factory import count_parameters, model_factory
from federated_intelligence.orchestration.local_executor import LocalExecutor
from federated_intelligence.server.coordinator import Coordinator
from federated_intelligence.utils.logging import get_logger
from federated_intelligence.utils.seeds import set_global_seed

logger = get_logger("federated_intelligence.runner")


def sanitize(obj: Any) -> Any:
    """Make a result JSON-compliant: non-finite floats (inf epsilon, NaN loss) become ``None``."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def generator_config(cfg: ExperimentConfig) -> GeneratorConfig:
    d = cfg.dataset
    return GeneratorConfig(
        seed=cfg.experiment.seed,
        num_clients=d.num_clients,
        num_features=d.num_features,
        min_samples_per_client=d.min_samples_per_client,
        max_samples_per_client=d.max_samples_per_client,
        scenario=Scenario(d.scenario),
        label_beta=d.label_beta,
        feature_shift=d.feature_shift,
        quantity_sigma=d.quantity_sigma,
        concept_noise=d.concept_noise,
        label_noise=d.label_noise,
        weight_scale=d.weight_scale,
        global_test_size=d.global_test_size,
    )


def build_dataset(cfg: ExperimentConfig) -> FederatedDataset:
    d = cfg.dataset
    if d.name in ("femnist", "shakespeare"):
        from federated_intelligence.data.loaders import load_femnist, load_shakespeare

        path = d.path or f"data/raw/{d.name}"
        loader = load_femnist if d.name == "femnist" else load_shakespeare
        return loader(path, seed=cfg.experiment.seed, **d.loader)
    gcfg = generator_config(cfg)
    if cfg.dataset.mode == "native":
        ds = generate_dataset(gcfg)
    else:
        kwargs = {"alpha": cfg.dataset.alpha} if cfg.dataset.partition == "dirichlet" else {}
        if cfg.dataset.partition == "quantity_skew":
            kwargs = {"sigma": cfg.dataset.quantity_sigma}
        ds = build_partitioned_dataset(gcfg, cfg.dataset.partition, cfg.dataset.total_samples, **kwargs)
    return ds.standardized()  # scaler fitted on training data only


def model_kwargs(cfg: ExperimentConfig, dataset: FederatedDataset) -> dict[str, Any]:
    """Constructor arguments of the configured model for this dataset."""
    name = cfg.model.name
    if name == "gru":
        return {"vocab_size": dataset.num_classes}
    if name == "cnn":
        return {"num_classes": dataset.num_classes}
    return {"num_features": dataset.num_features, "num_classes": dataset.num_classes, "hidden": cfg.model.hidden}


def build_executor(cfg: ExperimentConfig, dataset: FederatedDataset, model_fn, compressor):
    seed = cfg.experiment.seed
    if cfg.execution.backend == "ray":
        from federated_intelligence.orchestration.ray_executor import RayExecutor

        return RayExecutor(
            dataset.clients,
            model_fn,
            compressor,
            seed,
            num_workers=cfg.execution.num_workers,
            timeout_s=cfg.execution.client_timeout_s,
        )
    clients = {cid: FederatedClient(c, model_fn, compressor, seed) for cid, c in dataset.clients.items()}
    return LocalExecutor(clients)


def run_experiment(
    cfg: ExperimentConfig,
    out_dir: Path | None = None,
    tracker: Tracker | None = None,
    dataset: FederatedDataset | None = None,
    quiet: bool = True,
) -> dict[str, Any]:
    """Run one experiment and (optionally) persist ``<out_dir>/<name>__seed<seed>.json``."""
    seed = cfg.experiment.seed
    set_global_seed(seed)
    dataset = dataset or build_dataset(cfg)
    model_fn = model_factory(cfg.model.name, init_seed=seed, **model_kwargs(cfg, dataset))
    compressor = build_compressor(
        cfg.communication.compressor, cfg.communication.topk_ratio, cfg.communication.error_feedback
    )
    executor = build_executor(cfg, dataset, model_fn, compressor)
    tracker = tracker or build_tracker("none", None, cfg.experiment.name)
    tracker.start(cfg.experiment.name, cfg.model_dump(mode="json"))
    coordinator = Coordinator(
        cfg, dataset, model_fn, executor, compressor, callbacks=[tracker.log_round]
    )
    try:
        summary = coordinator.run()
        per_client = coordinator.per_client_final()
    finally:
        executor.shutdown()

    result: dict[str, Any] = {
        "name": cfg.experiment.name,
        "seed": seed,
        "config_hash": cfg.config_hash(),
        "config": cfg.model_dump(mode="json"),
        "dataset_metadata": {k: v for k, v in dataset.metadata.items() if k != "config"},
        "num_parameters": count_parameters(model_fn()),
        "summary": summary,
        "history": coordinator.history,
        "per_client_final": per_client,
        "environment": environment_info(),
        "compressor": compressor.describe(),
    }
    tracker.finish(summary)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{cfg.experiment.name}__seed{seed}.json"
        path.write_text(json.dumps(sanitize(result), default=float, allow_nan=False))
        result["result_path"] = str(path)
    if not quiet:
        s = summary["final"]
        logger.info("%s seed=%d test_acc=%.4f", cfg.experiment.name, seed, s["test_accuracy"])
    return result
