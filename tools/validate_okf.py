"""Validate an Open Knowledge Format bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlcourse.okf_validation import validate_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", default="okf", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict-warnings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_bundle(args.bundle, repository_root=Path.cwd())
    if args.format == "json":
        print(
            json.dumps(
                {
                    "concept_count": result.concept_count,
                    "index_count": result.index_count,
                    "diagnostics": [item.to_dict() for item in result.diagnostics],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for item in result.diagnostics:
            print(f"{item.severity.upper()} {item.code} {item.path}:{item.line} {item.message}")
        print(
            f"Validated {result.concept_count} concepts and {result.index_count} indexes: "
            f"{len(result.errors)} errors, {len(result.warnings)} warnings."
        )
    return int(bool(result.errors or (args.strict_warnings and result.warnings)))


if __name__ == "__main__":
    raise SystemExit(main())
