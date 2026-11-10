"""Payload serialization used for *measured* communication cost.

The size of a payload is always ``len(serialize(...))`` - the number of bytes that would
actually be sent - never an estimate based on tensor element counts.

Wire format ("FIB1", little-endian), deliberately compact so that header overhead does not
distort compression measurements on small models::

    magic(4) | n_arrays(u16) | repeat n: name_len(u8) name | dtype_len(u8) dtype |
                                          ndim(u8) shape(u32 * ndim) | nbytes(u64) | raw bytes

Arrays are read back with ``numpy.frombuffer`` (no pickle), so payloads are safe to parse.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence

import numpy as np

Params = list[np.ndarray]
_MAGIC = b"FIB1"


def serialize_arrays(arrays: Mapping[str, np.ndarray]) -> bytes:
    parts = [_MAGIC, struct.pack("<H", len(arrays))]
    for name, arr in arrays.items():
        a = np.asarray(arr)
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        nb, db = name.encode(), a.dtype.str.encode()
        parts.append(struct.pack("<B", len(nb)) + nb)
        parts.append(struct.pack("<B", len(db)) + db)
        parts.append(struct.pack("<B", a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape))
        data = a.tobytes()
        parts.append(struct.pack("<Q", len(data)) + data)
    return b"".join(parts)


def deserialize_arrays(payload: bytes) -> dict[str, np.ndarray]:
    if payload[:4] != _MAGIC:
        raise ValueError("not a FIB1 payload")
    (n,) = struct.unpack_from("<H", payload, 4)
    pos = 6
    out: dict[str, np.ndarray] = {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<B", payload, pos)
        pos += 1
        name = payload[pos : pos + ln].decode()
        pos += ln
        (ld,) = struct.unpack_from("<B", payload, pos)
        pos += 1
        dtype = np.dtype(payload[pos : pos + ld].decode())
        pos += ld
        (ndim,) = struct.unpack_from("<B", payload, pos)
        pos += 1
        shape = struct.unpack_from(f"<{ndim}I", payload, pos)
        pos += 4 * ndim
        (nbytes,) = struct.unpack_from("<Q", payload, pos)
        pos += 8
        if dtype.hasobject:
            raise ValueError("object dtypes are not allowed")
        out[name] = np.frombuffer(payload, dtype=dtype, count=nbytes // dtype.itemsize, offset=pos).reshape(shape).copy()
        pos += nbytes
    return out


def serialize_params(params: Sequence[np.ndarray]) -> bytes:
    return serialize_arrays({f"p{i:04d}": np.asarray(p) for i, p in enumerate(params)})


def deserialize_params(payload: bytes) -> Params:
    arrays = deserialize_arrays(payload)
    return [arrays[k] for k in sorted(arrays)]


def payload_size(params: Sequence[np.ndarray]) -> int:
    """Actual serialized size in bytes of a parameter list."""
    return len(serialize_params(params))
