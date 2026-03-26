"""Helpers for external dataset storage."""

from __future__ import annotations

from pathlib import Path

from mlcourse.paths import data_dir


def ensure_data_dir() -> Path:
    """Create the local dataset cache directory if needed and return it."""
    path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
