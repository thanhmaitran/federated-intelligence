"""Update compressors. Payload size is the real serialized byte length (``len(payload)``).

All compressors operate on a list of arrays (the model *update*, i.e. delta). Lossy compressors
support error feedback (EF): the part of the update that was not transmitted is stored in a
per-client residual and added to the next update, which preserves convergence for biased
compressors such as top-k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from federated_intelligence.communication.quantization import (
    dequantize_int8,
    from_float16,
    quantize_int8,
    to_float16,
)
from federated_intelligence.utils.serialization import deserialize_arrays, serialize_arrays

Params = list[np.ndarray]


def _flatten(params: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in params])


def _unflatten(flat: np.ndarray, template: Sequence[np.ndarray]) -> Params:
    out, pos = [], 0
    for t in template:
        out.append(flat[pos : pos + t.size].reshape(t.shape).astype(np.float32))
        pos += t.size
    return out


class Compressor(ABC):
    name = "base"
    lossy = False

    def __init__(self, error_feedback: bool = False):
        self.error_feedback = error_feedback and self.lossy

    @abstractmethod
    def _encode(self, flat: np.ndarray, template: Sequence[np.ndarray]) -> bytes: ...

    @abstractmethod
    def _decode(self, payload: bytes, template: Sequence[np.ndarray]) -> np.ndarray: ...

    def compress(
        self, update: Sequence[np.ndarray], state: dict | None = None
    ) -> bytes:
        flat = _flatten(update)
        if self.error_feedback and state is not None:
            flat = flat + state.get("residual", 0.0)
        payload = self._encode(flat, update)
        if self.error_feedback and state is not None:
            state["residual"] = flat - self._decode(payload, update)
        return payload

    def decompress(self, payload: bytes, template: Sequence[np.ndarray]) -> Params:
        return _unflatten(self._decode(payload, template), template)

    def describe(self) -> dict:
        return {"compressor": self.name, "error_feedback": self.error_feedback}


class NoCompression(Compressor):
    """Float32 baseline."""

    name = "float32"

    def _encode(self, flat, template):
        return serialize_arrays({"v": flat.astype(np.float32)})

    def _decode(self, payload, template):
        return deserialize_arrays(payload)["v"].astype(np.float32)


class Float16Compressor(Compressor):
    name = "float16"
    lossy = True

    def _encode(self, flat, template):
        return serialize_arrays({"v": to_float16(flat)})

    def _decode(self, payload, template):
        return from_float16(deserialize_arrays(payload)["v"])


class Int8Compressor(Compressor):
    """Per-tensor symmetric 8-bit quantization (one float32 scale per tensor)."""

    name = "int8"
    lossy = True

    def _encode(self, flat, template):
        arrays, pos = {}, 0
        for i, t in enumerate(template):
            seg = flat[pos : pos + t.size]
            q, s = quantize_int8(seg)
            arrays[f"q{i:04d}"] = q
            arrays[f"s{i:04d}"] = np.asarray(s, dtype=np.float32)
            pos += t.size
        return serialize_arrays(arrays)

    def _decode(self, payload, template):
        arrays = deserialize_arrays(payload)
        parts = [
            dequantize_int8(arrays[f"q{i:04d}"], arrays[f"s{i:04d}"]) for i in range(len(template))
        ]
        return np.concatenate(parts)


class TopKCompressor(Compressor):
    """Keep the ``ratio`` fraction of largest-magnitude coordinates (indices int32 + values f32)."""

    name = "topk"
    lossy = True

    def __init__(self, ratio: float = 0.1, error_feedback: bool = False):
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        super().__init__(error_feedback)
        self.ratio = ratio

    def _encode(self, flat, template):
        k = max(1, int(round(self.ratio * flat.size)))
        idx = np.argpartition(np.abs(flat), -k)[-k:] if k < flat.size else np.arange(flat.size)
        idx = np.sort(idx).astype(np.int32)
        return serialize_arrays(
            {
                "i": idx,
                "v": flat[idx].astype(np.float32),
                "n": np.asarray(flat.size, dtype=np.int64),
            }
        )

    def _decode(self, payload, template):
        a = deserialize_arrays(payload)
        out = np.zeros(int(a["n"]), dtype=np.float32)
        out[a["i"]] = a["v"]
        return out

    def describe(self) -> dict:
        return {**super().describe(), "ratio": self.ratio}


def build_compressor(name: str, ratio: float = 0.1, error_feedback: bool = False) -> Compressor:
    name = name.lower()
    if name in ("none", "float32"):
        return NoCompression()
    if name == "float16":
        return Float16Compressor(error_feedback)
    if name == "int8":
        return Int8Compressor(error_feedback)
    if name == "topk":
        return TopKCompressor(ratio, error_feedback)
    raise ValueError(f"Unknown compressor '{name}'")
