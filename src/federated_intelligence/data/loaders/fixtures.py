"""Tiny *fixture* datasets with the exact on-disk schema of FEMNIST and Shakespeare.

They let the loaders, the CNN/GRU pipelines and the CLI be exercised offline (CI, sandboxes
without access to Hugging Face). Fixtures are **random / toy data**: any number produced from them
is a wiring check and must never be reported as a FEMNIST or Shakespeare result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from federated_intelligence.data.loaders.shakespeare import ALL_LETTERS, SEQ_LEN


def make_femnist_fixture(
    path: str | Path, num_writers: int = 12, samples_per_writer: int = 60, num_classes: int = 62, seed: int = 0
) -> Path:
    """Write a Hugging Face Arrow dataset with columns image, writer_id, hsf_id, character.

    Each class is a fixed random 28x28 template plus noise, so a CNN can actually learn it.
    """
    from datasets import Dataset, Features, Image, Value
    from PIL import Image as PILImage

    rng = np.random.default_rng(seed)
    templates = rng.integers(0, 256, size=(num_classes, 28, 28))
    images, writers, hsf, chars = [], [], [], []
    for w in range(num_writers):
        style = rng.normal(0, 10, size=(28, 28))
        for _ in range(samples_per_writer):
            c = int(rng.integers(0, num_classes))
            arr = np.clip(templates[c] + style + rng.normal(0, 12, size=(28, 28)), 0, 255).astype(np.uint8)
            images.append(PILImage.fromarray(arr, mode="L"))
            writers.append(f"f{w:04d}")
            hsf.append(int(w % 4))
            chars.append(c)
    feats = Features(
        {"image": Image(), "writer_id": Value("string"), "hsf_id": Value("int64"), "character": Value("int64")}
    )
    ds = Dataset.from_dict(
        {"image": images, "writer_id": writers, "hsf_id": hsf, "character": chars}, features=feats
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    return path


def make_shakespeare_fixture(
    path: str | Path, num_roles: int = 6, windows_per_role: int = 1500, seed: int = 0
) -> Path:
    """Sliding 80-character windows over a per-role pseudo text (columns character_id, x, y)."""
    from datasets import Dataset

    rng = np.random.default_rng(seed)
    alphabet = ALL_LETTERS
    # A tiny Markov-ish text makes next-character prediction learnable above chance.
    trans = rng.dirichlet(np.ones(len(alphabet)) * 0.05, size=len(alphabet))
    cid, xs, ys = [], [], []
    for r in range(num_roles):
        n_chars = windows_per_role + SEQ_LEN + 1
        idx = [int(rng.integers(0, len(alphabet)))]
        for _ in range(n_chars - 1):
            idx.append(int(rng.choice(len(alphabet), p=trans[idx[-1]])))
        text = "".join(alphabet[i] for i in idx)
        for i in range(windows_per_role):
            cid.append(f"ROLE_{r}")
            xs.append(text[i : i + SEQ_LEN])
            ys.append(text[i + SEQ_LEN])
    ds = Dataset.from_dict({"character_id": cid, "x": xs, "y": ys})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    return path
