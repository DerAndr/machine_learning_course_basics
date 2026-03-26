"""Notebook-facing helpers."""

from __future__ import annotations

from pathlib import Path

from mlcourse.paths import project_root


def add_project_root_to_path() -> Path:
    """Return the project root for use from notebooks."""
    return project_root()
