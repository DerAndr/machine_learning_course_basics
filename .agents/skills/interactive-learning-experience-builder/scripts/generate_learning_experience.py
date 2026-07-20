import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

CONTENT_MARKER = "__CONTENT_JSON__"
STATIC_CONTENT_MARKER = "__STATIC_CONTENT__"
QUIZ_STATE_MACHINE_MARKER = "__QUIZ_STATE_MACHINE__"
VISUALIZATION_MODELS_MARKER = "__VISUALIZATION_MODELS__"
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
VISUALIZATION_TYPES = {
    "histogram",
    "boxplot",
    "scatter",
    "missingness",
    "binary-threshold",
    "labeled-scatter",
    "residual-diagnostics",
    "coefficient-path",
    "error-metrics",
}
SOURCE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:[^/].+")


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_missing(value: object) -> bool:
    return value is None or value == "" or value == []


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _has_valid_visualization_data(visualization_type: object, data: object) -> bool:
    if not isinstance(data, list) or not data:
        return False
    if not isinstance(visualization_type, str):
        return False
    if visualization_type in {"histogram", "boxplot"}:
        return all(_is_finite_number(value) for value in data)
    if visualization_type == "scatter":
        return all(
            isinstance(point, dict)
            and _is_finite_number(point.get("x"))
            and _is_finite_number(point.get("y"))
            for point in data
        )
    if visualization_type == "missingness":
        return all(
            isinstance(row, dict)
            and _is_non_empty_string(row.get("label"))
            and _is_finite_number(row.get("missing"))
            and _is_finite_number(row.get("total"))
            and row["total"] > 0
            and 0 <= row["missing"] <= row["total"]
            for row in data
        )
    return False


def _has_valid_histogram_bins(controls: object) -> bool:
    if controls is None:
        return True
    if not isinstance(controls, dict):
        return False
    bins = controls.get("bins")
    return (
        isinstance(bins, list)
        and bool(bins)
        and all(
            not isinstance(value, bool) and isinstance(value, int) and 1 <= value <= 50
            for value in bins
        )
    )


def _validate_binary_threshold(visualization: dict[str, object], location: str) -> list[str]:
    """Validate the probability-threshold teaching visualization."""
    errors: list[str] = []
    data = visualization.get("data")
    records_are_valid = isinstance(data, list) and len(data) >= 4
    if records_are_valid:
        ids: list[str] = []
        for record in data:
            if not isinstance(record, dict):
                records_are_valid = False
                break
            identifier = record.get("id")
            if not _is_non_empty_string(identifier):
                records_are_valid = False
                break
            ids.append(identifier)
            if (
                not _is_finite_number(record.get("score"))
                or not 0 <= record["score"] <= 1
                or isinstance(record.get("actual"), bool)
                or record.get("actual") not in (0, 1)
            ):
                records_are_valid = False
                break
        records_are_valid = records_are_valid and len(ids) == len(set(ids))
    if not records_are_valid:
        errors.append(
            f"{location}.data does not match the binary-threshold schema: "
            "records need unique IDs, scores from 0 through 1, and actual values 0 or 1"
        )

    controls = visualization.get("controls")
    controls_are_valid = isinstance(controls, dict) and all(
        _is_finite_number(controls.get(field))
        for field in ("minimum", "maximum", "step", "initial")
    )
    if controls_are_valid:
        minimum = controls["minimum"]
        maximum = controls["maximum"]
        step = controls["step"]
        initial = controls["initial"]
        controls_are_valid = (
            0 <= minimum <= initial <= maximum <= 1
            and minimum < maximum
            and 0 < step <= maximum - minimum
        )
    if not controls_are_valid:
        errors.append(
            f"{location}.controls does not match the binary-threshold schema: "
            "minimum, maximum, step, and initial must define a finite threshold range"
        )

    labels = visualization.get("labels")
    if not (
        isinstance(labels, dict)
        and _is_non_empty_string(labels.get("positive"))
        and _is_non_empty_string(labels.get("negative"))
    ):
        errors.append(
            f"{location}.labels does not match the binary-threshold schema: "
            "positive and negative labels must be non-empty"
        )
    return errors


def _validate_labeled_scatter(visualization: dict[str, object], location: str) -> list[str]:
    """Validate semantic groups and candidate boundaries."""
    errors: list[str] = []
    data = visualization.get("data")
    data_are_valid = isinstance(data, list) and len(data) >= 4
    point_ids: list[str] = []
    point_series: set[str] = set()
    if data_are_valid:
        for point in data:
            if not isinstance(point, dict):
                data_are_valid = False
                break
            identifier = point.get("id")
            series = point.get("series")
            if (
                not _is_non_empty_string(identifier)
                or not _is_non_empty_string(series)
                or not _is_finite_number(point.get("x"))
                or not _is_finite_number(point.get("y"))
            ):
                data_are_valid = False
                break
            point_ids.append(identifier)
            point_series.add(series)
        data_are_valid = data_are_valid and len(point_ids) == len(set(point_ids))

    labels = visualization.get("labels")
    labels_are_valid = (
        isinstance(labels, dict)
        and _is_non_empty_string(labels.get("x_axis"))
        and _is_non_empty_string(labels.get("y_axis"))
        and isinstance(labels.get("series"), dict)
        and len(labels["series"]) == 2
        and all(
            _is_non_empty_string(key) and _is_non_empty_string(value)
            for key, value in labels["series"].items()
        )
    )
    supported_series: set[str] = set()
    if labels_are_valid:
        supported_series = set(labels["series"])
        labels_are_valid = (
            len(point_series) == 2
            and point_series == supported_series
            and _is_non_empty_string(labels.get("positive_series"))
            and labels["positive_series"] in supported_series
        )
    if not data_are_valid or not labels_are_valid:
        errors.append(
            f"{location}.data does not match the labeled-scatter schema: "
            "points need unique IDs, finite coordinates, and exactly two labeled series"
        )
    if not labels_are_valid:
        errors.append(
            f"{location}.labels does not match the labeled-scatter schema: "
            "axis, series, and positive-series labels must be non-empty and cross-reference points"
        )

    controls = visualization.get("controls")
    controls_are_valid = isinstance(controls, dict) and isinstance(controls.get("boundaries"), list)
    boundary_ids: list[str] = []
    if controls_are_valid:
        boundaries = controls["boundaries"]
        controls_are_valid = bool(boundaries)
        for boundary in boundaries:
            if not isinstance(boundary, dict):
                controls_are_valid = False
                break
            identifier = boundary.get("id")
            if (
                not _is_non_empty_string(identifier)
                or not _is_non_empty_string(boundary.get("label"))
                or not _is_finite_number(boundary.get("slope"))
                or not _is_finite_number(boundary.get("intercept"))
            ):
                controls_are_valid = False
                break
            boundary_ids.append(identifier)
        controls_are_valid = (
            controls_are_valid
            and len(boundary_ids) == len(set(boundary_ids))
            and _is_non_empty_string(controls.get("initial"))
            and controls["initial"] in set(boundary_ids)
        )
    if not controls_are_valid:
        errors.append(
            f"{location}.controls does not match the labeled-scatter schema: "
            "boundaries need unique IDs, labels, finite lines, and a valid initial boundary"
        )
    return errors


def _validate_residual_diagnostics(visualization: dict[str, object], location: str) -> list[str]:
    """Validate precomputed fitted-value scenarios."""
    errors: list[str] = []
    data = visualization.get("data")
    scenarios_are_valid = isinstance(data, dict) and isinstance(data.get("scenarios"), list)
    scenario_ids: list[str] = []
    point_ids: list[str] = []
    if scenarios_are_valid:
        scenarios = data["scenarios"]
        scenarios_are_valid = bool(scenarios)
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                scenarios_are_valid = False
                break
            identifier = scenario.get("id")
            points = scenario.get("points")
            if (
                not _is_non_empty_string(identifier)
                or not _is_non_empty_string(scenario.get("label"))
                or not isinstance(points, list)
                or len(points) < 5
            ):
                scenarios_are_valid = False
                break
            scenario_ids.append(identifier)
            for point in points:
                if (
                    not isinstance(point, dict)
                    or not _is_non_empty_string(point.get("id"))
                    or not all(
                        _is_finite_number(point.get(field))
                        for field in ("x", "observed", "predicted")
                    )
                ):
                    scenarios_are_valid = False
                    break
                point_ids.append(point["id"])
            if not scenarios_are_valid:
                break
        scenarios_are_valid = (
            scenarios_are_valid
            and len(scenario_ids) == len(set(scenario_ids))
            and len(point_ids) == len(set(point_ids))
        )
    if not scenarios_are_valid:
        errors.append(
            f"{location}.data does not match the residual-diagnostics schema: "
            "scenarios need unique IDs and at least five finite, uniquely identified points"
        )

    controls = visualization.get("controls")
    if not (
        isinstance(controls, dict)
        and _is_non_empty_string(controls.get("initial"))
        and controls["initial"] in set(scenario_ids)
    ):
        errors.append(
            f"{location}.controls does not match the residual-diagnostics schema: "
            "initial must identify a scenario"
        )
    labels = visualization.get("labels")
    if not (
        isinstance(labels, dict)
        and all(
            _is_non_empty_string(labels.get(field))
            for field in ("x_axis", "target_axis", "residual_axis")
        )
    ):
        errors.append(
            f"{location}.labels does not match the residual-diagnostics schema: "
            "axis labels must be non-empty"
        )
    return errors


def _validate_coefficient_path(visualization: dict[str, object], location: str) -> list[str]:
    """Validate aligned Ridge and Lasso coefficient paths."""
    errors: list[str] = []
    data = visualization.get("data")
    data_are_valid = isinstance(data, dict)
    penalties: list[object] = []
    if data_are_valid:
        raw_penalties = data.get("penalties")
        raw_series = data.get("series")
        data_are_valid = (
            isinstance(raw_penalties, list)
            and len(raw_penalties) >= 3
            and isinstance(raw_series, list)
            and len(raw_series) >= 2
        )
        if data_are_valid:
            penalties = raw_penalties
            data_are_valid = all(
                _is_finite_number(value) and value >= 0 for value in penalties
            ) and all(left < right for left, right in zip(penalties, penalties[1:], strict=False))
            features: list[str] = []
            for series in raw_series:
                if not isinstance(series, dict) or not _is_non_empty_string(series.get("feature")):
                    data_are_valid = False
                    break
                features.append(series["feature"])
                for path in (series.get("ridge"), series.get("lasso")):
                    if (
                        not isinstance(path, list)
                        or len(path) != len(penalties)
                        or not all(_is_finite_number(value) for value in path)
                    ):
                        data_are_valid = False
                        break
                if not data_are_valid:
                    break
            data_are_valid = data_are_valid and len(features) == len(set(features))
    if not data_are_valid:
        errors.append(
            f"{location}.data does not match the coefficient-path schema: "
            "penalties and unique Ridge/Lasso feature paths must be finite and aligned"
        )

    controls = visualization.get("controls")
    if not (
        isinstance(controls, dict)
        and isinstance(controls.get("initial_index"), int)
        and not isinstance(controls.get("initial_index"), bool)
        and 0 <= controls["initial_index"] < len(penalties)
    ):
        errors.append(
            f"{location}.controls does not match the coefficient-path schema: "
            "initial_index must select one penalty"
        )
    return errors


def _validate_error_metrics(visualization: dict[str, object], location: str) -> list[str]:
    """Validate the fixed and adjustable errors used for metric comparison."""
    errors: list[str] = []
    data = visualization.get("data")
    data_are_valid = isinstance(data, dict)
    adjustable_errors: list[object] = []
    if data_are_valid:
        base_errors = data.get("base_errors")
        raw_adjustable_errors = data.get("adjustable_error")
        data_are_valid = (
            isinstance(base_errors, list)
            and len(base_errors) >= 3
            and all(_is_finite_number(error) for error in base_errors)
            and isinstance(raw_adjustable_errors, list)
            and len(raw_adjustable_errors) >= 3
            and all(_is_finite_number(error) and error >= 0 for error in raw_adjustable_errors)
        )
        if data_are_valid:
            adjustable_errors = raw_adjustable_errors
            data_are_valid = all(
                left < right
                for left, right in zip(adjustable_errors, adjustable_errors[1:], strict=False)
            )
    if not data_are_valid:
        errors.append(
            f"{location}.data does not match the error-metrics schema: "
            "base errors and increasing non-negative adjustable errors must be finite"
        )

    controls = visualization.get("controls")
    if not (
        isinstance(controls, dict)
        and isinstance(controls.get("initial_index"), int)
        and not isinstance(controls.get("initial_index"), bool)
        and 0 <= controls["initial_index"] < len(adjustable_errors)
    ):
        errors.append(
            f"{location}.controls does not match the error-metrics schema: "
            "initial_index must select one adjustable error"
        )
    labels = visualization.get("labels")
    if not (isinstance(labels, dict) and _is_non_empty_string(labels.get("units"))):
        errors.append(
            f"{location}.labels does not match the error-metrics schema: units must be non-empty"
        )
    return errors


def validate_visualization(visualization: dict[str, object], location: str) -> list[str]:
    """Return schema violations for a supported visualization."""
    visualization_type = visualization.get("type")
    validators = {
        "binary-threshold": _validate_binary_threshold,
        "labeled-scatter": _validate_labeled_scatter,
        "residual-diagnostics": _validate_residual_diagnostics,
        "coefficient-path": _validate_coefficient_path,
        "error-metrics": _validate_error_metrics,
    }
    validator = validators.get(visualization_type) if isinstance(visualization_type, str) else None
    if validator is not None:
        return validator(visualization, location)
    errors: list[str] = []
    if not _has_valid_visualization_data(visualization_type, visualization.get("data")):
        errors.append(f"{location}.data does not match the {visualization_type} schema")
    if visualization_type == "histogram" and not _has_valid_histogram_bins(
        visualization.get("controls")
    ):
        errors.append(f"{location}.controls.bins must contain positive integers up to 50")
    return errors


def _is_repository_relative_path(value: object) -> bool:
    if not _is_non_empty_string(value):
        return False
    source = value.strip()
    if "\\" in source or ":" in source or "?" in source or "#" in source:
        return False
    if source.startswith("/") or re.match(r"^[A-Za-z]:", source):
        return False
    parts = source.split("/")
    return not any(part in {"", ".", ".."} for part in parts)


def _is_named_source(value: object) -> bool:
    if not _is_non_empty_string(value):
        return False
    source = value.strip()
    return (
        _is_repository_relative_path(source)
        or source.startswith(("http://", "https://"))
        or bool(SOURCE_IDENTIFIER.fullmatch(source))
    )


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


def _validate_source_files(payload: dict[str, object], repository_root: Path | None) -> list[str]:
    if repository_root is None:
        return []
    errors: list[str] = []
    for source in sorted(set(_payload_sources(payload))):
        if _is_repository_relative_path(source) and not (repository_root / source).is_file():
            errors.append(f"source file does not exist: {source}")
    return errors


def validate_payload(payload: dict[str, object], repository_root: Path | None = None) -> list[str]:
    """Return every content-contract violation in a portable experience payload."""
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
        for field in ("experience_id", "title"):
            if not _is_non_empty_string(meta.get(field)):
                errors.append(f"meta.{field} must be a non-empty string")
        sources = meta.get("sources")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(_is_named_source(source) for source in sources)
        ):
            errors.append("meta.sources must contain named repository paths, URLs, or identifiers")

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
                or not all(_is_named_source(source) for source in sources)
            ):
                errors.append(
                    f"{location}.sources must contain named repository paths, URLs, or identifiers"
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
            if (
                not isinstance(visualization_type, str)
                or visualization_type not in VISUALIZATION_TYPES
            ):
                errors.append(
                    f"{location}.type {visualization_type!r} is unsupported; "
                    f"use one of {', '.join(sorted(VISUALIZATION_TYPES))}"
                )
            for field in ("id", "title", "explanation"):
                if not _is_non_empty_string(visualization.get(field)):
                    errors.append(f"{location}.{field} must be a non-empty string")
            errors.extend(validate_visualization(visualization, location))
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
                question_type = question.get("type")
                if not isinstance(question_type, str) or question_type not in QUESTION_TYPES:
                    errors.append(f"{location}.type is unsupported")
                for field in ("prompt", "explanation", "concept"):
                    if not _is_non_empty_string(question.get(field)):
                        errors.append(f"{location}.{field} must be a non-empty string")
                options = question.get("options")
                options_are_readable = isinstance(options, list) and all(
                    _is_non_empty_string(option) for option in options
                )
                if not isinstance(options, list):
                    errors.append(f"{location}.options must be an array")
                elif not options_are_readable:
                    errors.append(f"{location}.options must contain readable strings")
                elif (
                    isinstance(question_type, str)
                    and question_type in {"single-choice", "multiple-choice"}
                    and not options
                ):
                    errors.append(f"{location}.options must contain choices for this question type")
                answer = question.get("answer")
                if _is_missing(answer):
                    errors.append(f"{location}.answer is required")
                elif question_type == "single-choice" and (
                    not isinstance(answer, str)
                    or not isinstance(options, list)
                    or answer not in options
                ):
                    errors.append(f"{location}.answer must be one available option")
                elif question_type == "multiple-choice" and (
                    not isinstance(answer, list)
                    or not answer
                    or not all(_is_non_empty_string(item) for item in answer)
                ):
                    errors.append(
                        f"{location}.answer must be a non-empty subset of available options"
                    )
                elif question_type == "multiple-choice" and len(answer) != len(set(answer)):
                    errors.append(f"{location}.answer choices must be unique")
                elif question_type == "multiple-choice" and (
                    not options_are_readable or not set(answer).issubset(set(options))
                ):
                    errors.append(
                        f"{location}.answer must be a non-empty subset of available options"
                    )
                elif question_type == "interpretation":
                    if not _is_non_empty_string(answer):
                        errors.append(f"{location}.answer must be a readable interpretation")
                    elif options_are_readable and options and answer not in options:
                        errors.append(f"{location}.answer must be one available option")

        extra_levels = sorted(quizzes.keys() - set(QUIZ_LEVELS))
        if extra_levels:
            errors.append(f"unsupported quiz levels: {', '.join(extra_levels)}")

    break_prompts = payload.get("break_prompts")
    if (
        not isinstance(break_prompts, list)
        or not break_prompts
        or not all(_is_non_empty_string(prompt) for prompt in break_prompts)
    ):
        errors.append("break_prompts must always embed at least one readable prompt")

    errors.extend(_validate_source_files(payload, repository_root))
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
        "<h2>Complete static learning reference</h2>",
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
    sections.append("</section><section><h3>Sources</h3><ul>")
    meta = payload["meta"]
    assert isinstance(meta, dict)
    for source in meta["sources"]:
        sections.append(f"<li>{escape(source)}</li>")
    sections.append("</ul></section></div>")
    return "".join(sections)


def render_site(
    template: str,
    payload: dict[str, object],
    quiz_state_machine: str | None = None,
    visualization_models: str | None = None,
) -> str:
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
    quiz_marker_count = template.count(QUIZ_STATE_MACHINE_MARKER)
    if quiz_marker_count > 1:
        raise ValueError(
            "template must contain no more than one "
            f"{QUIZ_STATE_MACHINE_MARKER} marker; found {quiz_marker_count}"
        )
    if quiz_marker_count == 1 and quiz_state_machine is None:
        raise ValueError("template requires the embedded quiz state machine")
    model_marker_count = template.count(VISUALIZATION_MODELS_MARKER)
    if model_marker_count > 1:
        raise ValueError(
            "template must contain no more than one "
            f"{VISUALIZATION_MODELS_MARKER} marker; found {model_marker_count}"
        )
    if model_marker_count == 1 and visualization_models is None:
        raise ValueError("template requires the embedded visualization models")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    static_content = _render_static_content(payload)
    rendered = template.replace(STATIC_CONTENT_MARKER, static_content).replace(
        CONTENT_MARKER, encoded
    )
    if quiz_marker_count == 1:
        assert quiz_state_machine is not None
        rendered = rendered.replace(QUIZ_STATE_MACHINE_MARKER, quiz_state_machine)
    if model_marker_count == 1:
        assert visualization_models is not None
        rendered = rendered.replace(VISUALIZATION_MODELS_MARKER, visualization_models)
    return rendered


def generate_site(
    payload: dict[str, object],
    template: str,
    quiz_state_machine: str | None = None,
    visualization_models: str | None = None,
) -> str:
    """Validate payload content and return one portable HTML document."""
    errors = validate_payload(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return render_site(template, payload, quiz_state_machine, visualization_models)


def write_site(
    content_path: Path,
    template_path: Path,
    output_path: Path,
    repository_root: Path | None = None,
) -> Path:
    """Validate content and write one deterministic, portable HTML file."""
    raw_payload: Any = json.loads(content_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("content JSON must contain one top-level object")
    payload: dict[str, object] = raw_payload
    errors = validate_payload(payload, repository_root=repository_root)
    if errors:
        raise ValueError("\n".join(errors))

    template = template_path.read_text(encoding="utf-8")
    quiz_state_machine = None
    if QUIZ_STATE_MACHINE_MARKER in template:
        quiz_state_machine = template_path.with_name("quiz-state-machine.js").read_text(
            encoding="utf-8"
        )
    visualization_models = None
    if VISUALIZATION_MODELS_MARKER in template:
        visualization_models = template_path.with_name("visualization-models.js").read_text(
            encoding="utf-8"
        )
    html = generate_site(payload, template, quiz_state_machine, visualization_models)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8", newline="\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one offline interactive learning experience."
    )
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        output_path = write_site(
            args.content,
            args.template,
            args.output,
            repository_root=Path.cwd(),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        for line in str(error).splitlines():
            print(f"ERROR: {line}")
        return 1
    print(f"GENERATED: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
