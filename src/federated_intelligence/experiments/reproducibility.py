"""Reproducibility metadata: environment, git revision, config hash."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "scipy": _version("scipy"),
        "scikit-learn": _version("scikit-learn"),
        "ray": _version("ray"),
        "git_revision": git_revision(),
    }
