import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

RESTRICTED_SOURCE_TOKENS = {
    "gradebook",
    "grading",
    "private",
    "quiz",
    "quizzes",
    "solution",
    "solutions",
    "teacher",
}


def _load_core_generator() -> ModuleType:
    skills_root = Path(__file__).resolve().parents[2]
    script = (
        skills_root
        / "interactive-learning-experience-builder"
        / "scripts"
        / "generate_learning_experience.py"
    )
    spec = importlib.util.spec_from_file_location("portable_learning_experience_generator", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load portable generator: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload_sources(payload: dict[str, object]) -> list[str]:
    sources: list[str] = []
    meta = payload.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("sources"), list):
        sources.extend(source for source in meta["sources"] if isinstance(source, str))
    concepts = payload.get("concepts")
    if isinstance(concepts, list):
        for concept in concepts:
            if isinstance(concept, dict) and isinstance(concept.get("sources"), list):
                sources.extend(source for source in concept["sources"] if isinstance(source, str))
    return sources


def _restricted_source_token(source: str) -> str | None:
    tokens = set(re.findall(r"[a-z0-9]+", source.casefold()))
    restricted = sorted(tokens & RESTRICTED_SOURCE_TOKENS)
    if restricted:
        return restricted[0]
    if {"answer", "key"}.issubset(tokens) or {"answer", "keys"}.issubset(tokens):
        return "answer key"
    return None


def _is_canonical_repository_path(source: str) -> bool:
    if (
        source != source.strip()
        or "\\" in source
        or any(mark in source for mark in (":", "?", "#"))
    ):
        return False
    if source.startswith("/"):
        return False
    parts = source.split("/")
    return not any(part in {"", ".", ".."} for part in parts)


def _source_allowlist_error(source: str, lecture_slug: str) -> str | None:
    selected_lecture_prefix = f"lectures/{lecture_slug}/"
    if source.startswith(selected_lecture_prefix) or source.startswith("okf/"):
        if _is_canonical_repository_path(source):
            return None
        return f"source is not a canonical repository-relative path: {source}"
    if source.startswith("lectures/"):
        return f"source belongs to a different lecture than {lecture_slug}: {source}"
    if source.casefold().startswith(("lectures/", "okf/")):
        return f"source root must use canonical lowercase spelling: {source}"
    return (
        "source is outside the ML-course allowlist; use only "
        f"{selected_lecture_prefix}... or okf/...: {source}"
    )


def validate_course_source_policy(
    payload: dict[str, object],
    lecture_slug: str,
) -> list[str]:
    """Allow only the selected lecture and read-only OKF sources."""
    if not re.fullmatch(r"[a-z0-9_]+", lecture_slug):
        return [f"selected lecture identifier is not canonical lowercase: {lecture_slug}"]

    errors: list[str] = []
    for source in sorted(set(_payload_sources(payload))):
        allowlist_error = _source_allowlist_error(source, lecture_slug)
        if allowlist_error:
            errors.append(allowlist_error)
            continue
        restricted_token = _restricted_source_token(source)
        if restricted_token:
            errors.append(f"restricted course source ({restricted_token}) is not allowed: {source}")
    return errors


def generate_course_site(
    content_path: Path,
    template_path: Path,
    output_path: Path,
    lecture_slug: str,
    repository_root: Path,
) -> Path:
    """Apply ML-course source policy, then delegate rendering to the portable core."""
    raw_payload = json.loads(content_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("content JSON must contain one top-level object")
    errors = validate_course_source_policy(raw_payload, lecture_slug)
    if errors:
        raise ValueError("\n".join(errors))

    generator = _load_core_generator()
    return generator.write_site(
        content_path,
        template_path,
        output_path,
        repository_root=repository_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an ML-course experience after enforcing course source policy."
    )
    parser.add_argument("--lecture-slug", required=True)
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        output_path = generate_course_site(
            args.content,
            args.template,
            args.output,
            args.lecture_slug,
            repository_root=Path.cwd(),
        )
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as error:
        for line in str(error).splitlines():
            print(f"ERROR: {line}")
        return 1
    print(f"GENERATED: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
