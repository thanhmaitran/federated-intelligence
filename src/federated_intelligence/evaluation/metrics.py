"""Evaluation metrics shared by centralized baselines and federated experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


@torch.no_grad()
def predict_logits(
    model: nn.Module, x: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i : i + batch_size])
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 0), dtype=np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Accuracy, loss, precision/recall/F1 (macro) and, for binary tasks, ROC-AUC / PR-AUC."""
    probs = _softmax(logits)
    pred = probs.argmax(axis=1)
    loss = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())
    out = {
        "loss": loss,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    if probs.shape[1] == 2 and len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, probs[:, 1]))
        out["pr_auc"] = float(average_precision_score(y, probs[:, 1]))
    return out


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return classification_metrics(predict_logits(model, x), y)


def perplexity(loss: float) -> float:
    """Perplexity for language-modelling cross-entropy (natural log)."""
    return float(np.exp(loss))
