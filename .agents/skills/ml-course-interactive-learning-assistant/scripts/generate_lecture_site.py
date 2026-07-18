import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

CONTENT_MARKER = "__CONTENT_JSON__"
STATIC_CONTENT_MARKER = "__STATIC_CONTENT__"
REQUIRED_TOP_LEVEL = {
    "meta",
    "defaults",
    "concepts",
    "visualizations",
    "quizzes",
    "break_prompts",
}
QUIZ_LEVELS = ("foundations", "applied", "challenge")
QUESTION_FIELDS = {
    "id",
    "type",
    "prompt",
    "options",
    "answer",
    "explanation",
    "concept",
}
QUESTION_TYPES = {"single-choice", "multiple-choice", "interpretation"}
VISUALIZATION_TYPES = {"histogram", "boxplot", "scatter", "missingness"}
PRIVATE_SOURCE_PARTS = {
    "answer_key",
    "answer_keys",
    "gradebook",
    "grading",
    "private",
    "quiz",
    "quizzes",
    "solution",
    "solutions",
}


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_missing(value: object) -> bool:
    return value is None or value == "" or value == []


def _is_public_source_path(value: object) -> bool:
    if not _is_non_empty_string(value):
        return False
    source = value.strip()
    if "\\" in source or "://" in source or "?" in source or "#" in source:
        return False
    if source.startswith("/") or re.match(r"^[A-Za-z]:", source):
        return False
    parts = source.split("/")
    if parts[0] not in {"lectures", "okf"}:
        return False
    if any(part in {"", ".", ".."} for part in parts):
        return False
    normalized_parts = {part.lower().replace("-", "_") for part in parts}
    return normalized_parts.isdisjoint(PRIVATE_SOURCE_PARTS)


def validate_payload(payload: dict[str, object]) -> list[str]:
    """Return every content-contract violation in a lecture payload."""
    errors: list[str] = []

    missing_keys = sorted(REQUIRED_TOP_LEVEL - payload.keys())
    extra_keys = sorted(payload.keys() - REQUIRED_TOP_LEVEL)
    if missing_keys:
        errors.append(f"missing top-level keys: {', '.join(missing_keys)}")
    if extra_keys:
        errors.append(f"unsupported top-level keys: {', '.join(extra_keys)}")

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
    else:
        for field in ("lecture_slug", "title"):
            if not _is_non_empty_string(meta.get(field)):
                errors.append(f"meta.{field} must be a non-empty string")
        sources = meta.get("sources")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(_is_public_source_path(source) for source in sources)
        ):
            errors.append(
                "meta.sources must contain public repository-relative paths under lectures/ or okf/"
            )

    defaults = payload.get("defaults")
    if not isinstance(defaults, dict):
        errors.append("defaults must be an object")
    else:
        if defaults.get("difficulty") not in QUIZ_LEVELS:
            errors.append("defaults.difficulty must be foundations, applied, or challenge")
        for field in ("focus_mode", "color_blind", "break_prompts"):
            if not isinstance(defaults.get(field), bool):
                errors.append(f"defaults.{field} must be a boolean")

    concepts = payload.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        errors.append("concepts must be a non-empty array")
    else:
        for index, concept in enumerate(concepts):
            location = f"concepts[{index}]"
            if not isinstance(concept, dict):
                errors.append(f"{location} must be an object")
                continue
            for field in ("id", "title", "explanation", "interpretation"):
                if not _is_non_empty_string(concept.get(field)):
                    errors.append(f"{location}.{field} must be a non-empty string")
            mistakes = concept.get("common_mistakes")
            if (
                not isinstance(mistakes, list)
                or not mistakes
                or not all(_is_non_empty_string(item) for item in mistakes)
            ):
                errors.append(f"{location}.common_mistakes must contain readable mistakes")
            sources = concept.get("sources")
            if (
                not isinstance(sources, list)
                or not sources
                or not all(_is_public_source_path(source) for source in sources)
            ):
                errors.append(
                    f"{location}.sources must contain public repository-relative "
                    "paths under lectures/ or okf/"
                )

    visualizations = payload.get("visualizations")
    if not isinstance(visualizations, list) or not visualizations:
        errors.append("visualizations must be a non-empty array")
    else:
        for index, visualization in enumerate(visualizations):
            location = f"visualizations[{index}]"
            if not isinstance(visualization, dict):
                errors.append(f"{location} must be an object")
                continue
            visualization_type = visualization.get("type")
            if visualization_type not in VISUALIZATION_TYPES:
                errors.append(
                    f"{location}.type {visualization_type!r} is unsupported; "
                    f"use one of {', '.join(sorted(VISUALIZATION_TYPES))}"
                )
            for field in ("id", "title", "explanation"):
                if not _is_non_empty_string(visualization.get(field)):
                    errors.append(f"{location}.{field} must be a non-empty string")
            if "data" not in visualization:
                errors.append(f"{location}.data is required")
            if not _is_non_empty_string(visualization.get("fallback")):
                errors.append(f"{location}.fallback must be readable without a graph")

    quizzes = payload.get("quizzes")
    question_ids: set[str] = set()
    if not isinstance(quizzes, dict):
        errors.append("quizzes must be an object")
    else:
        for level in QUIZ_LEVELS:
            questions = quizzes.get(level)
            if not isinstance(questions, list):
                errors.append(f"quizzes.{level} must be an array of exactly 10 questions")
                continue
            if len(questions) != 10:
                errors.append(
                    f"quizzes.{level} must contain exactly 10 questions; found {len(questions)}"
                )
            for index, question in enumerate(questions):
                location = f"quizzes.{level}[{index}]"
                if not isinstance(question, dict):
                    errors.append(f"{location} must be an object")
                    continue
                missing_fields = QUESTION_FIELDS - question.keys()
                for field in sorted(missing_fields):
                    errors.append(f"{location}.{field} is required")
                question_id = question.get("id")
                if not _is_non_empty_string(question_id):
                    errors.append(f"{location}.id must be a non-empty string")
                elif question_id in question_ids:
                    errors.append(f"{location}.id {question_id!r} is not unique")
                else:
                    question_ids.add(question_id)
                if question.get("type") not in QUESTION_TYPES:
                    errors.append(f"{location}.type is unsupported")
                for field in ("prompt", "explanation", "concept"):
                    if not _is_non_empty_string(question.get(field)):
                        errors.append(f"{location}.{field} must be a non-empty string")
                if not isinstance(question.get("options"), list):
                    errors.append(f"{location}.options must be an array")
                if _is_missing(question.get("answer")):
                    errors.append(f"{location}.answer is required")

        extra_levels = sorted(quizzes.keys() - set(QUIZ_LEVELS))
        if extra_levels:
            errors.append(f"unsupported quiz levels: {', '.join(extra_levels)}")

    break_prompts = payload.get("break_prompts")
    if (
        not isinstance(break_prompts, list)
        or not break_prompts
        or not all(_is_non_empty_string(prompt) for prompt in break_prompts)
    ):
        errors.append("break_prompts must always embed at least one readable lecture prompt")

    return errors


def _static_text(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _render_static_content(payload: dict[str, object]) -> str:
    """Render the complete learning reference without relying on JavaScript."""

    def escape(value: object) -> str:
        return html.escape(_static_text(value), quote=False)

    sections: list[str] = [
        '<div class="static-reference-body">',
        "<h2>Complete static lecture reference</h2>",
        (
            "<p>This reference remains available when scripts or browser storage "
            "are unavailable.</p>"
        ),
        "<section><h3>Concepts</h3>",
    ]
    for concept in payload["concepts"]:
        assert isinstance(concept, dict)
        sections.extend(
            [
                "<article>",
                f"<h4>{escape(concept['title'])}</h4>",
                f"<p>{escape(concept['explanation'])}</p>",
                f"<p><strong>How to interpret it:</strong> {escape(concept['interpretation'])}</p>",
                "<h5>Common mistakes</h5><ul>",
            ]
        )
        for mistake in concept["common_mistakes"]:
            sections.append(f"<li>{escape(mistake)}</li>")
        sections.extend(["</ul><p><strong>Sources:</strong> "])
        sections.append("; ".join(escape(source) for source in concept["sources"]))
        sections.append("</p></article>")
    sections.append("</section><section><h3>Visualization reference</h3>")
    for visualization in payload["visualizations"]:
        assert isinstance(visualization, dict)
        sections.extend(
            [
                "<article>",
                f"<h4>{escape(visualization['title'])}</h4>",
                f"<p>{escape(visualization['explanation'])}</p>",
                (
                    '<p class="graph-fallback"><strong>Graph fallback:</strong> '
                    f"{escape(visualization['fallback'])}</p>"
                ),
                "</article>",
            ]
        )
    sections.append("</section><section><h3>Quiz banks and answer review</h3>")
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    for level in QUIZ_LEVELS:
        sections.append(f"<section><h4>{escape(level.title())}</h4><ol>")
        questions = quizzes[level]
        assert isinstance(questions, list)
        for question in questions:
            assert isinstance(question, dict)
            sections.extend(
                [
                    "<li>",
                    f"<p><strong>{escape(question['prompt'])}</strong></p>",
                ]
            )
            options = question["options"]
            assert isinstance(options, list)
            if options:
                sections.append("<ul>")
                for option in options:
                    sections.append(f"<li>{escape(option)}</li>")
                sections.append("</ul>")
            sections.extend(
                [
                    f"<p><strong>Answer:</strong> {escape(question['answer'])}</p>",
                    (f"<p><strong>Explanation:</strong> {escape(question['explanation'])}</p>"),
                    "</li>",
                ]
            )
        sections.append("</ol></section>")
    sections.append("</section><section><h3>Course sources</h3><ul>")
    meta = payload["meta"]
    assert isinstance(meta, dict)
    for source in meta["sources"]:
        sections.append(f"<li>{escape(source)}</li>")
    sections.append("</ul></section></div>")
    return "".join(sections)


def render_site(template: str, payload: dict[str, object]) -> str:
    """Embed deterministic JSON and a complete escaped static reference."""
    marker_count = template.count(CONTENT_MARKER)
    if marker_count != 1:
        raise ValueError(
            f"template must contain exactly one {CONTENT_MARKER} marker; found {marker_count}"
        )
    static_marker_count = template.count(STATIC_CONTENT_MARKER)
    if static_marker_count != 1:
        raise ValueError(
            "template must contain exactly one "
            f"{STATIC_CONTENT_MARKER} marker; found {static_marker_count}"
        )
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    static_content = _render_static_content(payload)
    return template.replace(STATIC_CONTENT_MARKER, static_content).replace(
        CONTENT_MARKER,
        encoded,
    )


def generate_site(
    content_path: Path,
    template_path: Path,
    output_path: Path,
) -> Path:
    """Validate content and write one deterministic, portable HTML file."""
    raw_payload: Any = json.loads(content_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("content JSON must contain one top-level object")
    payload: dict[str, object] = raw_payload
    errors = validate_payload(payload)
    if errors:
        raise ValueError("\n".join(errors))

    template = template_path.read_text(encoding="utf-8")
    html = render_site(template, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one offline interactive lecture review site."
    )
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        output_path = generate_site(args.content, args.template, args.output)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        for line in str(error).splitlines():
            print(f"ERROR: {line}")
        return 1
    print(f"GENERATED: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
