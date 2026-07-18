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
        "renderHistogram",
        "renderBoxplot",
        "renderScatter",
        "renderMissingness",
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
  <label>Break prompts <input data-setting="break-prompts" type="checkbox"></label>
  <main id="main-content">
    <section class="panel progress-panel">Progress</section>
    <section class="graph-fallback">The same lesson is available as text.</section>
  </main>
  <noscript><section id="static-content">Static learning content.</section></noscript>
</body>
</html>
"""


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
