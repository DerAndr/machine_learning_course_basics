# Machine Learning Course Authoring Repository

This repository is the private authoring source of truth for the course materials.

## Purpose

- Maintain lecture assignments and solutions in English.
- Keep notebooks versionable via Jupytext pairs.
- Prepare future publication into a separate student-facing read-only repository.
- Keep datasets outside git and reference them via metadata and helper utilities.

## Repository Layout

- `lectures/`: lecture materials and metadata.
- `src/mlcourse/`: reusable Python helpers shared across notebooks.
- `docs/`: repository workflow and publishing notes.
- `publish/lectures.yaml`: release-state manifest for future student exports.
- `legacy_import/`: temporary landing zone for raw legacy materials.

## Canonical Commands

```bash
uv sync
uv run jupyter lab
uv run ruff format .
uv run ruff check .
uv run ty check src tests
```

## Migration Sequence

1. Initialize the scaffold and tooling.
2. Import legacy course materials verbatim into `legacy_import/`.
3. Refactor imported materials into the normalized lecture structure.
