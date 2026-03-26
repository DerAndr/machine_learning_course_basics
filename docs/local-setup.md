# Local Setup

## Requirements

- Python 3.12
- `uv`

## Initial Setup

```bash
uv sync
```

## Common Commands

Start JupyterLab:

```bash
uv run jupyter lab
```

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Run type checks:

```bash
uv run ty check src tests
```

Run tests:

```bash
uv run pytest
```
