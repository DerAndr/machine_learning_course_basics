"""Path helpers for course materials."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def lectures_dir() -> Path:
    """Return the lectures directory."""
    return project_root() / "lectures"


def data_dir() -> Path:
    """Return the local data cache directory."""
    return project_root() / ".data"
