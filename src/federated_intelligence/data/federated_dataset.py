"""In-memory container for federated datasets and per-client train/val/test splits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClientSplit:
    """Local data of one client. Raw arrays never leave the client in simulation."""

    client_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def num_train(self) -> int:
        return len(self.y_train)

    @property
    def num_samples(self) -> int:
        return len(self.y_train) + len(self.y_val) + len(self.y_test)


@dataclass
class FederatedDataset:
    name: str
    clients: dict[str, ClientSplit]
    global_test_x: np.ndarray
    global_test_y: np.ndarray
    num_features: int
    num_classes: int
    metadata: dict = field(default_factory=dict)
    # Additional named evaluation sets, e.g. {"unseen_writers": (x, y)} for FEMNIST.
    extra_test_sets: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def client_ids(self) -> list[str]:
        return sorted(self.clients)

    def pooled_train(self) -> tuple[np.ndarray, np.ndarray]:
        xs = [self.clients[c].x_train for c in self.client_ids]
        ys = [self.clients[c].y_train for c in self.client_ids]
        return np.concatenate(xs), np.concatenate(ys)

    def pooled_val(self) -> tuple[np.ndarray, np.ndarray]:
        xs = [self.clients[c].x_val for c in self.client_ids]
        ys = [self.clients[c].y_val for c in self.client_ids]
        return np.concatenate(xs), np.concatenate(ys)

    def standardized(self) -> FederatedDataset:
        """Return a copy standardized with statistics fitted on *training* data only.

        Statistics come from the union of client training sets (a one-off federated
        preprocessing step); validation/test data never influence the scaler.
        """
        x_tr, _ = self.pooled_train()
        mu = x_tr.mean(axis=0)
        sd = x_tr.std(axis=0)
        sd[sd < 1e-8] = 1.0

        def f(a: np.ndarray) -> np.ndarray:
            return ((a - mu) / sd).astype(np.float32)

        clients = {
            cid: ClientSplit(
                cid,
                f(c.x_train),
                c.y_train,
                f(c.x_val),
                c.y_val,
                f(c.x_test),
                c.y_test,
            )
            for cid, c in self.clients.items()
        }
        meta = dict(self.metadata)
        meta["standardized"] = True
        return FederatedDataset(
            self.name,
            clients,
            f(self.global_test_x),
            self.global_test_y,
            self.num_features,
            self.num_classes,
            meta,
            {k: (f(x), y) for k, (x, y) in self.extra_test_sets.items()},
        )


def split_indices(
    n: int, fractions: tuple[float, float, float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random train/val/test index split with at least one sample in each part."""
    perm = rng.permutation(n)
    n_train = max(1, int(round(n * fractions[0])))
    n_val = max(1, int(round(n * fractions[1])))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]
