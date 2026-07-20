import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".agents" / "skills" / "interactive-learning-experience-builder" / "scripts"
TEMPLATE = (
    ROOT
    / ".agents"
    / "skills"
    / "interactive-learning-experience-builder"
    / "assets"
    / "learning-experience-template.html"
)


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_script("generate_learning_experience")
validator = _load_script("validate_learning_experience")
generate_site = generator.generate_site
render_site = generator.render_site
validate_payload = generator.validate_payload
validate_html = validator.validate_html


LEVELS = ("foundations", "applied", "challenge")


SEMANTIC_VISUALIZATIONS = {
    "binary-threshold": {
        "id": "threshold",
        "type": "binary-threshold",
        "title": "Threshold",
        "explanation": "Thresholds convert scores to decisions.",
        "data": [
            {"id": "p1", "score": 0.9, "actual": 1},
            {"id": "p2", "score": 0.6, "actual": 0},
            {"id": "p3", "score": 0.4, "actual": 1},
            {"id": "p4", "score": 0.1, "actual": 0},
        ],
        "controls": {"minimum": 0.1, "maximum": 0.9, "step": 0.1, "initial": 0.5},
        "labels": {"positive": "Positive", "negative": "Negative"},
        "fallback": "At 0.5 there is one TP, one FP, one TN, and one FN.",
    },
    "labeled-scatter": {
        "id": "boundary",
        "type": "labeled-scatter",
        "title": "Boundary",
        "explanation": "A line divides two illustrative classes.",
        "data": [
            {"id": "a1", "x": 1, "y": 1, "series": "A"},
            {"id": "a2", "x": 2, "y": 1.5, "series": "A"},
            {"id": "b1", "x": 3, "y": 3, "series": "B"},
            {"id": "b2", "x": 4, "y": 4, "series": "B"},
        ],
        "controls": {
            "boundaries": [
                {
                    "id": "balanced",
                    "label": "Balanced boundary",
                    "slope": -1,
                    "intercept": 5,
                }
            ],
            "initial": "balanced",
        },
        "labels": {
            "x_axis": "Feature 1",
            "y_axis": "Feature 2",
            "series": {"A": "Class A", "B": "Class B"},
            "positive_series": "B",
        },
        "fallback": "Class A occupies the lower-left and Class B the upper-right.",
    },
    "residual-diagnostics": {
        "id": "residuals",
        "type": "residual-diagnostics",
        "title": "Residuals",
        "explanation": "Residual patterns reveal missed structure.",
        "data": {
            "scenarios": [
                {
                    "id": "appropriate",
                    "label": "Appropriate fit",
                    "points": [
                        {
                            "id": f"r{index}",
                            "x": index,
                            "observed": 2 * index + 1,
                            "predicted": 2 * index + 0.9,
                        }
                        for index in range(1, 6)
                    ],
                }
            ]
        },
        "controls": {"initial": "appropriate"},
        "labels": {
            "x_axis": "Predictor",
            "target_axis": "Observed target",
            "residual_axis": "Residual",
        },
        "fallback": "Residuals stay close to zero without a systematic pattern.",
    },
    "coefficient-path": {
        "id": "coefficients",
        "type": "coefficient-path",
        "title": "Coefficient paths",
        "explanation": "Ridge shrinks while Lasso can reach zero.",
        "data": {
            "penalties": [0, 0.5, 1],
            "series": [
                {"feature": "Area", "ridge": [3, 2, 1], "lasso": [3, 1.5, 0]},
                {"feature": "Age", "ridge": [-1, -0.7, -0.4], "lasso": [-1, -0.4, 0]},
            ],
        },
        "controls": {"initial_index": 0},
        "fallback": "Ridge values shrink; both Lasso values reach zero at penalty 1.",
    },
    "error-metrics": {
        "id": "metrics",
        "type": "error-metrics",
        "title": "Metric sensitivity",
        "explanation": "RMSE reacts more strongly to a large error.",
        "data": {"base_errors": [-2, -1, 0, 1, 2], "adjustable_error": [0, 5, 10]},
        "controls": {"initial_index": 1},
        "labels": {"units": "target units"},
        "fallback": "Increasing one error from 0 to 10 raises RMSE faster than MAE.",
    },
}


def _question(level: str, number: int) -> dict[str, object]:
    return {
        "id": f"{level}-{number}",
        "type": "single-choice",
        "prompt": f"Question {number}?",
        "options": ["A", "B"],
        "answer": "A",
        "explanation": "A is supported by the lecture source.",
        "concept": "distribution",
    }


@pytest.fixture
def payload() -> dict[str, object]:
    return {
        "meta": {
            "experience_id": "distribution-basics",
            "title": "Distribution Basics",
            "sources": ["knowledge/distributions.md"],
        },
        "defaults": {
            "difficulty": "foundations",
            "focus_mode": True,
            "color_blind": True,
            "break_prompts": False,
        },
        "concepts": [
            {
                "id": "distribution",
                "title": "Distribution",
                "explanation": "A distribution describes how values vary.",
                "interpretation": "Inspect shape, centre, and spread together.",
                "common_mistakes": ["Treating one summary as the whole distribution."],
                "sources": ["knowledge/distributions.md"],
            }
        ],
        "visualizations": [
            {
                "id": "distribution-shape",
                "type": "histogram",
                "title": "Distribution shape",
                "explanation": "Bin width changes the visible shape.",
                "data": [1, 2, 2, 3],
                "controls": {"bins": [2, 4]},
                "fallback": "Values cluster near 2; the table is 1, 2, 2, 3.",
            }
        ],
        "quizzes": {
            level: [_question(level, number) for number in range(1, 11)] for level in LEVELS
        },
        "break_prompts": ["Stretch those whiskers before the next box plot."],
    }


def test_render_site_embeds_deterministic_compact_json(
    payload: dict[str, object],
) -> None:
    html = render_site(
        ("<main>__STATIC_CONTENT__</main><script>const CONTENT = __CONTENT_JSON__;</script>"),
        payload,
    )

    assert "__CONTENT_JSON__" not in html
    assert "__STATIC_CONTENT__" not in html
    expected = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    expected = (
        expected.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    assert expected in html


def test_render_site_escapes_script_closing_sequences_in_embedded_json(
    payload: dict[str, object],
) -> None:
    concepts = payload["concepts"]
    assert isinstance(concepts, list)
    concept = concepts[0]
    assert isinstance(concept, dict)
    concept["explanation"] = "</script><script>alert('unsafe')</script>"

    rendered = render_site(
        "<main>__STATIC_CONTENT__</main><script>const CONTENT = __CONTENT_JSON__;</script>",
        payload,
    )
    script = rendered.split("<script>const CONTENT = ", maxsplit=1)[1]

    assert "\\u003c/script\\u003e" in script
    assert "<script>alert('unsafe')" not in script


@pytest.mark.parametrize("marker_count", [0, 2])
def test_render_site_requires_exactly_one_content_marker(
    payload: dict[str, object],
    marker_count: int,
) -> None:
    template = "__CONTENT_JSON__".join(["<p></p>"] * (marker_count + 1))

    with pytest.raises(ValueError, match="exactly one"):
        render_site(template, payload)


@pytest.mark.parametrize("marker_count", [0, 2])
def test_render_site_requires_exactly_one_static_content_marker(
    payload: dict[str, object],
    marker_count: int,
) -> None:
    static = "__STATIC_CONTENT__".join(["<p></p>"] * (marker_count + 1))
    template = f"{static}<script>const CONTENT = __CONTENT_JSON__;</script>"

    with pytest.raises(ValueError, match="exactly one"):
        render_site(template, payload)


def test_render_site_embeds_visualization_models_and_requires_the_source(
    payload: dict[str, object],
) -> None:
    template = (
        "<main>__STATIC_CONTENT__</main>"
        "<script>const CONTENT = __CONTENT_JSON__;</script>"
        "<script>__VISUALIZATION_MODELS__</script>"
    )

    with pytest.raises(ValueError, match="visualization models"):
        render_site(template, payload)

    rendered = render_site(template, payload, visualization_models="const models = {};")

    assert rendered.count("const models = {};") == 1
    assert "__VISUALIZATION_MODELS__" not in rendered


def test_render_site_rejects_multiple_visualization_model_markers(
    payload: dict[str, object],
) -> None:
    template = (
        "<main>__STATIC_CONTENT__</main>"
        "<script>const CONTENT = __CONTENT_JSON__;</script>"
        "<script>__VISUALIZATION_MODELS____VISUALIZATION_MODELS__</script>"
    )

    with pytest.raises(ValueError, match="no more than one"):
        render_site(template, payload, visualization_models="const models = {};")


def test_render_site_embeds_escaped_complete_static_reference(
    payload: dict[str, object],
) -> None:
    concepts = payload["concepts"]
    quizzes = payload["quizzes"]
    assert isinstance(concepts, list)
    assert isinstance(concepts[0], dict)
    assert isinstance(quizzes, dict)
    concepts[0]["explanation"] = 'Shape <script>alert("no")</script> & spread.'
    foundations = quizzes["foundations"]
    assert isinstance(foundations, list)
    foundations[0]["prompt"] = "Which value is < 3?"
    foundations[0]["options"] = ["A & B", "<three>"]
    foundations[0]["answer"] = "<three>"

    html = render_site(
        "<article>__STATIC_CONTENT__</article><script>const CONTENT = __CONTENT_JSON__;</script>",
        payload,
    )
    static_html = html.split("<script>const CONTENT =", maxsplit=1)[0]

    assert 'Shape &lt;script&gt;alert("no")&lt;/script&gt; &amp; spread.' in static_html
    assert "Which value is &lt; 3?" in static_html
    assert "A &amp; B" in static_html
    assert "&lt;three&gt;" in static_html
    for level in LEVELS:
        questions = quizzes[level]
        assert isinstance(questions, list)
        for question in questions:
            assert isinstance(question, dict)
            for field in ("prompt", "answer", "explanation"):
                assert (
                    str(question[field])
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    in static_html
                )
            options = question["options"]
            assert isinstance(options, list)
            for option in options:
                assert (
                    str(option).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    in static_html
                )


def test_validate_payload_reports_all_content_problems(
    payload: dict[str, object],
) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    foundations = quizzes["foundations"]
    applied = quizzes["applied"]
    assert isinstance(foundations, list)
    assert isinstance(applied, list)
    foundations.pop()
    applied[0].pop("explanation")

    visualizations = payload["visualizations"]
    assert isinstance(visualizations, list)
    visualizations[0].pop("fallback")
    visualizations.append(
        {
            "id": "unsupported",
            "type": "heatmap",
            "title": "Unsupported",
            "explanation": "Not part of the contract.",
            "data": [],
            "fallback": "Text fallback.",
        }
    )

    errors = validate_payload(payload)

    assert any("foundations" in error and "10" in error for error in errors)
    assert any("explanation" in error for error in errors)
    assert any("fallback" in error for error in errors)
    assert any("heatmap" in error for error in errors)


def test_validate_payload_requires_embedded_break_prompts(
    payload: dict[str, object],
) -> None:
    payload["break_prompts"] = []

    errors = validate_payload(payload)

    assert any("break_prompts" in error for error in errors)


@pytest.mark.parametrize(
    ("visualization_type", "data"),
    [
        ("histogram", []),
        ("boxplot", [1, "not-a-number"]),
        ("scatter", [{"x": 1}]),
        ("missingness", [{"label": "age", "missing": 3, "total": 0}]),
    ],
)
def test_validate_payload_rejects_invalid_visualization_data(
    payload: dict[str, object],
    visualization_type: str,
    data: object,
) -> None:
    visualizations = payload["visualizations"]
    assert isinstance(visualizations, list)
    visualization = visualizations[0]
    assert isinstance(visualization, dict)
    visualization["type"] = visualization_type
    visualization["data"] = data

    errors = validate_payload(payload)

    assert any("data" in error for error in errors)


@pytest.mark.parametrize("visualization", SEMANTIC_VISUALIZATIONS.values())
def test_validate_payload_accepts_semantic_visualization(
    payload: dict[str, object], visualization: dict[str, object]
) -> None:
    payload["visualizations"] = [visualization]

    assert validate_payload(payload) == []


@pytest.mark.parametrize(
    ("visualization_type", "field_path"),
    [
        ("binary-threshold", ("data", 0, "actual")),
        ("labeled-scatter", ("data", 0, "series")),
        ("residual-diagnostics", ("data", "scenarios", 0, "points")),
        ("coefficient-path", ("data", "series", 0, "lasso")),
        ("error-metrics", ("data", "adjustable_error")),
    ],
)
def test_validate_payload_rejects_broken_semantic_schema(
    payload: dict[str, object],
    visualization_type: str,
    field_path: tuple[object, ...],
) -> None:
    visualization = copy.deepcopy(SEMANTIC_VISUALIZATIONS[visualization_type])
    target: object = visualization
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    del target[field_path[-1]]  # type: ignore[index]
    payload["visualizations"] = [visualization]

    errors = validate_payload(payload)

    assert any(visualization_type in error for error in errors)


def test_validate_payload_rejects_choice_question_without_options(
    payload: dict[str, object],
) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    questions = quizzes["foundations"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    question["options"] = []

    errors = validate_payload(payload)

    assert any("options" in error for error in errors)


@pytest.mark.parametrize("bins", [[0, 4], ["many"], []])
def test_validate_payload_rejects_invalid_histogram_bins(
    payload: dict[str, object],
    bins: list[object],
) -> None:
    visualizations = payload["visualizations"]
    assert isinstance(visualizations, list)
    visualization = visualizations[0]
    assert isinstance(visualization, dict)
    visualization["controls"] = {"bins": bins}

    errors = validate_payload(payload)

    assert any("controls.bins" in error for error in errors)


@pytest.mark.parametrize(
    ("question_type", "answer"),
    [
        ("single-choice", "C"),
        ("single-choice", ["A"]),
        ("multiple-choice", "A"),
        ("multiple-choice", ["A", "C"]),
    ],
)
def test_validate_payload_rejects_impossible_choice_answers(
    payload: dict[str, object],
    question_type: str,
    answer: object,
) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    questions = quizzes["foundations"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    question["type"] = question_type
    question["answer"] = answer

    errors = validate_payload(payload)

    assert any("answer" in error for error in errors)


def test_validate_payload_rejects_duplicate_multiple_choice_answers(
    payload: dict[str, object],
) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    questions = quizzes["foundations"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    question["type"] = "multiple-choice"
    question["answer"] = ["A", "A"]

    errors = validate_payload(payload)

    assert any("answer" in error and "unique" in error for error in errors)


def test_validate_payload_reports_non_string_multiple_choice_options_without_crashing(
    payload: dict[str, object],
) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    questions = quizzes["foundations"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    question["type"] = "multiple-choice"
    question["options"] = ["A", {"label": "B"}]
    question["answer"] = ["A"]

    errors = validate_payload(payload)

    assert any("options must contain readable strings" in error for error in errors)


def test_validate_payload_requires_option_based_interpretation_answer_to_be_available(
    payload: dict[str, object],
) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    questions = quizzes["foundations"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    question["type"] = "interpretation"
    question["answer"] = "A different interpretation"

    errors = validate_payload(payload)

    assert any("answer must be one available option" in error for error in errors)


@pytest.mark.parametrize(
    "source",
    ["/knowledge/source.md", "C:/knowledge/source.md", "../knowledge/source.md", ""],
)
def test_validate_payload_rejects_unsafe_meta_source_paths(
    payload: dict[str, object],
    source: str,
) -> None:
    meta = payload["meta"]
    assert isinstance(meta, dict)
    meta["sources"] = [source]

    errors = validate_payload(payload)

    assert any("meta.sources" in error for error in errors)


@pytest.mark.parametrize(
    "source",
    ["/knowledge/source.md", "C:/knowledge/source.md", "../knowledge/source.md", ""],
)
def test_validate_payload_rejects_unsafe_concept_source_paths(
    payload: dict[str, object],
    source: str,
) -> None:
    concepts = payload["concepts"]
    assert isinstance(concepts, list)
    concept = concepts[0]
    assert isinstance(concept, dict)
    concept["sources"] = [source]

    errors = validate_payload(payload)

    assert any("concepts[0].sources" in error for error in errors)


def test_validate_payload_checks_only_repository_relative_source_files(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    meta = payload["meta"]
    assert isinstance(meta, dict)
    meta["sources"] = ["knowledge/not-a-real-file.md", "https://example.test/source", "kb:item"]
    concepts = payload["concepts"]
    assert isinstance(concepts, list)
    concept = concepts[0]
    assert isinstance(concept, dict)
    concept["sources"] = meta["sources"]

    errors = validate_payload(payload, repository_root=tmp_path)

    assert errors == ["source file does not exist: knowledge/not-a-real-file.md"]


def test_generate_site_rejects_invalid_payload(payload: dict[str, object]) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    challenge = quizzes["challenge"]
    assert isinstance(challenge, list)
    challenge.pop()
    with pytest.raises(ValueError, match="challenge"):
        generate_site(payload, "<main>__STATIC_CONTENT__</main><script>__CONTENT_JSON__</script>")


def test_generate_site_returns_one_portable_html_document(payload: dict[str, object]) -> None:
    result = generate_site(
        payload,
        "<!doctype html><main>__STATIC_CONTENT__</main>"
        "<script>const CONTENT = __CONTENT_JSON__;</script>",
    )

    assert result.startswith("<!doctype html>")
    assert "__CONTENT_JSON__" not in result
    assert "__STATIC_CONTENT__" not in result


def test_minimal_template_exposes_offline_accessibility_contract() -> None:
    template_path = (
        ROOT
        / ".agents"
        / "skills"
        / "interactive-learning-experience-builder"
        / "assets"
        / "learning-experience-template.html"
    )

    template = template_path.read_text(encoding="utf-8")

    assert template.count("__CONTENT_JSON__") == 1
    assert template.count("__STATIC_CONTENT__") == 1
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in template
    assert '<main id="main-content">' in template
    assert '<section id="static-content"' in template
    assert "<noscript>" in template
    assert "<script>const CONTENT = __CONTENT_JSON__;</script>" in template
    assert ":focus-visible" in template
    assert "prefers-reduced-motion" in template


@pytest.mark.parametrize(
    "hook",
    [
        'data-setting="difficulty"',
        'data-setting="focus"',
        'data-setting="color-blind"',
        'data-setting="break-prompts"',
        "__VISUALIZATION_MODELS__",
        'id="palette-status"',
        "--graph-primary: #6d28d9",
        "--graph-secondary: #0f766e",
        "--graph-primary: #0072b2",
        "--graph-secondary: #d55e00",
        "renderHistogram",
        "renderBoxplot",
        "renderScatter",
        "renderMissingness",
        "renderBinaryThreshold",
        "renderLabeledScatter",
        "renderResidualDiagnostics",
        "renderCoefficientPath",
        "renderErrorMetrics",
        "LearningVisualizationModels.thresholdSummary",
        "LearningVisualizationModels.seriesStyles",
        "renderQuiz",
        "showQuizResults",
        "safeStorage",
        'aria-live="polite"',
    ],
)
def test_interactive_template_exposes_stable_behavior_hooks(hook: str) -> None:
    template_path = (
        ROOT
        / ".agents"
        / "skills"
        / "interactive-learning-experience-builder"
        / "assets"
        / "learning-experience-template.html"
    )

    template = template_path.read_text(encoding="utf-8")

    assert hook in template


def test_palette_modes_use_visibly_different_graph_tokens() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "--graph-primary: #6d28d9" in template
    assert "--graph-secondary: #0f766e" in template
    assert "--graph-primary: #0072b2" in template
    assert "--graph-secondary: #d55e00" in template
    assert "Palette: standard" in template
    assert "Palette: color-blind-safe" in template


def test_semantic_renderers_use_payload_labels_and_non_color_cues() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    for hook in (
        "visualization.labels.positive",
        "visualization.labels.negative",
        "True series",
        "predicted as",
        "Positive residual",
        "Negative residual",
        "coefficientFeatureStyle",
        "feature-label",
        "stroke-dasharray",
    ):
        assert hook in template


def test_interactive_template_preserves_non_color_and_focus_friendly_cues() -> None:
    template_path = (
        ROOT
        / ".agents"
        / "skills"
        / "interactive-learning-experience-builder"
        / "assets"
        / "learning-experience-template.html"
    )
    template = template_path.read_text(encoding="utf-8")

    assert "<pattern" in template
    assert 'data-shape="' in template
    assert 'class="graph-fallback"' in template
    assert 'data-focus-friendly="' in template
    assert "document.activeElement" in template
    assert "confirm(" in template
    assert 'id="break-prompt"' in template
    assert 'id="quiz-feedback"' in template
    assert 'id="quiz-results"' in template
    assert 'id="retry-quiz"' in template


@pytest.mark.parametrize(
    "hook",
    [
        'data-viz-control="',
        'class="chart-summary"',
        'id="previous-visualization"',
        'id="next-visualization"',
        "visualizationIndex",
        'id="interpretation-answer"',
        'tabindex="-1"',
        "validDifficulties",
        'typeof parsed === "object"',
        'byId("next-question").focus()',
    ],
)
def test_interactive_template_covers_reviewed_runtime_paths(hook: str) -> None:
    template_path = (
        ROOT
        / ".agents"
        / "skills"
        / "interactive-learning-experience-builder"
        / "assets"
        / "learning-experience-template.html"
    )

    assert hook in template_path.read_text(encoding="utf-8")


def test_quiz_retry_state_contract() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    for hook in (
        "firstAttemptCorrect",
        "attempts",
        "clearQuizFeedback",
        "disableQuestionInputs",
        "if (!correct)",
        "totalAttempts",
    ):
        assert hook in template


def test_correct_answer_scrolls_focused_next_control_into_view() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    expected_sequence = """byId("next-question").focus();
      byId("next-question").scrollIntoView({ block: "nearest" });"""

    assert expected_sequence in template


def test_template_has_mobile_sticky_progress_contract() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    for hook in (
        'class="panel progress-panel"',
        "position: sticky",
        "env(safe-area-inset-top",
        "scroll-padding-top",
        "@media (max-width:",
    ):
        assert hook in template


def _valid_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html { scroll-padding-top: 4rem; }
    .progress-panel { position: sticky; }
    :focus-visible { outline: 3px solid currentColor; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation: none; transition: none; }
    }
  </style>
</head>
<body>
  <label>Difficulty <select data-setting="difficulty"></select></label>
  <label>Focus <input data-setting="focus" type="checkbox"></label>
  <label>Colour blind <input data-setting="color-blind" type="checkbox"></label>
  <p id="palette-status">Palette: standard</p>
  <label>Break prompts <input data-setting="break-prompts" type="checkbox"></label>
  <main id="main-content">
    <section class="panel progress-panel">Progress</section>
    <section class="graph-fallback">The same lesson is available as text.</section>
  </main>
  <noscript><section id="static-content">Static learning content.</section></noscript>
  <script>globalThis.LearningVisualizationModels = {};</script>
</body>
</html>
"""


def test_validate_html_requires_palette_status(tmp_path: Path) -> None:
    path = tmp_path / "index.html"
    path.write_text(_valid_html().replace('id="palette-status"', ""), encoding="utf-8")
    assert any("palette-status" in error for error in validate_html(path))


def test_validate_html_rejects_duplicate_palette_status(tmp_path: Path) -> None:
    path = tmp_path / "index.html"
    duplicate = '<p id="palette-status">Palette: standard</p>'
    path.write_text(
        _valid_html().replace("</body>", f"{duplicate}</body>"),
        encoding="utf-8",
    )

    assert "missing live palette status" in validate_html(path)


def test_validate_html_requires_embedded_visualization_models(tmp_path: Path) -> None:
    path = tmp_path / "index.html"
    path.write_text(
        _valid_html().replace(
            "<script>globalThis.LearningVisualizationModels = {};</script>",
            "",
        ),
        encoding="utf-8",
    )

    assert "missing embedded visualization models" in validate_html(path)


def test_validate_html_does_not_mistake_model_usage_for_embedded_models(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.html"
    path.write_text(
        _valid_html().replace(
            "globalThis.LearningVisualizationModels = {};",
            "LearningVisualizationModels.thresholdSummary([], 0.5);",
        ),
        encoding="utf-8",
    )

    assert "missing embedded visualization models" in validate_html(path)


@pytest.mark.parametrize(
    "model_text",
    [
        "// globalThis.LearningVisualizationModels = {};",
        'const note = "globalThis.LearningVisualizationModels = {};";',
        "const note = `globalThis.LearningVisualizationModels = {};`;",
    ],
)
def test_validate_html_rejects_comment_or_string_model_assignment(
    tmp_path: Path,
    model_text: str,
) -> None:
    path = tmp_path / "index.html"
    path.write_text(
        _valid_html().replace(
            "globalThis.LearningVisualizationModels = {};",
            model_text,
        ),
        encoding="utf-8",
    )

    assert "missing embedded visualization models" in validate_html(path)


def test_validate_html_rejects_unreplaced_visualization_model_marker(tmp_path: Path) -> None:
    path = tmp_path / "index.html"
    path.write_text(
        _valid_html().replace(
            "globalThis.LearningVisualizationModels = {};",
            "__VISUALIZATION_MODELS__",
        ),
        encoding="utf-8",
    )

    assert "unreplaced visualization model marker" in validate_html(path)


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ('class="panel progress-panel"', "progress panel"),
        ("position: sticky", "sticky progress"),
    ],
)
def test_validate_html_requires_sticky_progress_contract(
    tmp_path: Path,
    fragment: str,
    expected: str,
) -> None:
    path = tmp_path / "index.html"
    path.write_text(_valid_html().replace(fragment, ""), encoding="utf-8")

    errors = validate_html(path)

    assert any(expected in error.lower() for error in errors)


def test_validate_html_rejects_sticky_progress_declaration_inside_css_comment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.html"
    path.write_text(
        _valid_html().replace(
            ".progress-panel { position: sticky; }",
            "/* .progress-panel { position: sticky; } */",
        ),
        encoding="utf-8",
    )

    errors = validate_html(path)

    assert "missing sticky progress style" in errors


@pytest.mark.parametrize(
    "resource",
    [
        '<script src="https://example.test/app.js"></script>',
        '<link rel="stylesheet" href="https://example.test/app.css">',
        '<img src="https://example.test/chart.png" alt="Chart">',
        '<link rel="preload" as="font" href="https://example.test/font.woff2">',
        '<script src="./app.js"></script>',
        '<link rel="stylesheet" href="./app.css">',
        '<img src="./chart.png" alt="Chart">',
        '<source src="./chart.svg">',
        '<link rel="preload" as="font" href="./font.woff2">',
    ],
)
def test_validate_html_rejects_external_runtime_resources(
    tmp_path: Path,
    resource: str,
) -> None:
    path = tmp_path / "index.html"
    path.write_text(_valid_html().replace("</head>", f"{resource}</head>"))

    errors = validate_html(path)

    assert any("external" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "style_dependency",
    [
        '@import "./theme.css";',
        '.chart { background-image: url("./chart.png"); }',
    ],
)
def test_validate_html_rejects_local_css_runtime_dependencies(
    tmp_path: Path,
    style_dependency: str,
) -> None:
    path = tmp_path / "index.html"
    path.write_text(
        _valid_html().replace("</style>", f"{style_dependency}</style>"),
        encoding="utf-8",
    )

    errors = validate_html(path)

    assert any("style" in error.lower() and "not portable" in error.lower() for error in errors)


def test_validate_html_allows_embedded_and_fragment_css_urls(tmp_path: Path) -> None:
    path = tmp_path / "index.html"
    css = """
    .embedded { background-image: url("data:image/svg+xml,%3Csvg%3E%3C/svg%3E"); }
    .patterned { fill: url("#diagonal-hatch"); }
    """
    path.write_text(
        _valid_html().replace("</style>", f"{css}</style>"),
        encoding="utf-8",
    )

    assert validate_html(path) == []


@pytest.mark.parametrize(
    "runtime_code",
    [
        'fetch("https://example.test/data.json")',
        "new XMLHttpRequest()",
        'new WebSocket("wss://example.test/socket")',
        'new EventSource("https://example.test/events")',
        'navigator.sendBeacon("https://example.test/progress", "done")',
        'import("./lecture-chunk.js")',
        'document.createElement("script")',
        "element.src = 'https://example.test/chart.png'",
    ],
)
def test_validate_html_rejects_inline_network_capabilities(
    tmp_path: Path,
    runtime_code: str,
) -> None:
    path = tmp_path / "index.html"
    html = _valid_html().replace(
        "</body>",
        f"<script>{runtime_code}</script></body>",
    )
    path.write_text(html, encoding="utf-8")

    errors = validate_html(path)

    assert any("network-capable" in error for error in errors)


def test_validate_html_allows_network_api_words_inside_content_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.html"
    html = _valid_html().replace(
        "</body>",
        (
            '<script>const CONTENT = {"explanation": '
            '"The word fetch() is inert lecture text."};</script></body>'
        ),
    )
    path.write_text(html, encoding="utf-8")

    assert validate_html(path) == []


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ('data-setting="difficulty"', "difficulty"),
        ("<noscript>", "noscript"),
        ("prefers-reduced-motion", "reduced"),
        (":focus-visible", "focus"),
        ('class="graph-fallback"', "fallback"),
    ],
)
def test_validate_html_rejects_missing_required_contract(
    tmp_path: Path,
    fragment: str,
    expected: str,
) -> None:
    path = tmp_path / "index.html"
    path.write_text(_valid_html().replace(fragment, ""), encoding="utf-8")

    errors = validate_html(path)

    assert any(expected in error.lower() for error in errors)


@pytest.mark.parametrize("tag", ["div", "span"])
def test_validate_html_rejects_inert_settings_hooks(
    tmp_path: Path,
    tag: str,
) -> None:
    path = tmp_path / "index.html"
    html = _valid_html().replace(
        '<select data-setting="difficulty"></select>',
        f'<{tag} data-setting="difficulty">Foundations</{tag}>',
    )
    path.write_text(html, encoding="utf-8")

    errors = validate_html(path)

    assert any("difficulty" in error for error in errors)


def test_validate_html_accepts_keyboard_operable_settings_role(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.html"
    html = _valid_html().replace(
        '<select data-setting="difficulty"></select>',
        (
            '<div data-setting="difficulty" role="combobox" tabindex="0" '
            'onkeydown="chooseDifficulty(event)">Foundations</div>'
        ),
    )
    path.write_text(html, encoding="utf-8")

    assert validate_html(path) == []


@pytest.mark.parametrize(
    "fallback",
    [
        '<section class="graph-fallback"></section>',
        '<section class="graph-fallback"> \n\t </section>',
    ],
)
def test_validate_html_rejects_empty_graph_fallback(
    tmp_path: Path,
    fallback: str,
) -> None:
    path = tmp_path / "index.html"
    html = _valid_html().replace(
        ('<section class="graph-fallback">The same lesson is available as text.</section>'),
        fallback,
    )
    path.write_text(html, encoding="utf-8")

    errors = validate_html(path)

    assert any("fallback" in error for error in errors)


def test_validate_html_reports_every_missing_setting(tmp_path: Path) -> None:
    html = _valid_html()
    for setting in ("difficulty", "focus", "color-blind", "break-prompts"):
        html = html.replace(f'data-setting="{setting}"', "")
    path = tmp_path / "index.html"
    path.write_text(html, encoding="utf-8")

    errors = validate_html(path)

    assert all(
        any(setting in error for error in errors)
        for setting in ("difficulty", "focus", "color-blind", "break-prompts")
    )
