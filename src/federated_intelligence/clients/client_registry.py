"""Registry of the clients known to the coordinator (metadata only, no data)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClientInfo:
    client_id: str
    num_train_samples: int


class ClientRegistry:
    def __init__(self) -> None:
        self._clients: dict[str, ClientInfo] = {}

    def register(self, info: ClientInfo) -> None:
        if info.client_id in self._clients:
            raise ValueError(f"client {info.client_id} already registered")
        self._clients[info.client_id] = info

    def __len__(self) -> int:
        return len(self._clients)

    @property
    def ids(self) -> list[str]:
        return sorted(self._clients)

    def info(self, client_id: str) -> ClientInfo:
        return self._clients[client_id]

    def num_samples(self, client_id: str) -> int:
        return self._clients[client_id].num_train_samples
