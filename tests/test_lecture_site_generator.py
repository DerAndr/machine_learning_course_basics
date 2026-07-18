import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".agents" / "skills" / "ml-course-interactive-learning-assistant" / "scripts"


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_script("generate_lecture_site")
validator = _load_script("validate_lecture_site")
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
            "lecture_slug": "lecture_01_eda",
            "title": "Exploratory Data Analysis",
            "sources": ["lectures/lecture_01_eda/lecture_notes.md"],
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
                "sources": ["lectures/lecture_01_eda/lecture_notes.md"],
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
        "<script>const CONTENT = __CONTENT_JSON__;</script>",
        payload,
    )

    assert "__CONTENT_JSON__" not in html
    assert (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        in html
    )


@pytest.mark.parametrize("marker_count", [0, 2])
def test_render_site_requires_exactly_one_content_marker(
    payload: dict[str, object],
    marker_count: int,
) -> None:
    template = "__CONTENT_JSON__".join(["<p></p>"] * (marker_count + 1))

    with pytest.raises(ValueError, match="exactly one"):
        render_site(template, payload)


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
    "source",
    [
        "https://example.test/lecture-notes",
        "/lectures/lecture_01_eda/lecture_notes.md",
        "C:/lectures/lecture_01_eda/lecture_notes.md",
        "../lectures/lecture_01_eda/lecture_notes.md",
        "lectures/lecture_01_eda/../private_notes.md",
        "lectures/lecture_01_eda/answer_keys/solutions.md",
        "lectures/lecture_01_eda/quizzes/questions.json",
        "lectures/lecture_01_eda/private/draft.md",
        "lectures/lecture_01_eda/solution/walkthrough.md",
        "lectures/lecture_01_eda/solutions/walkthrough.md",
        "lectures/lecture_01_eda/grading/rubric.md",
        "lectures/lecture_01_eda/gradebook/scores.csv",
    ],
)
def test_validate_payload_rejects_non_public_source_paths(
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
    [
        "https://example.test/concept",
        "lectures/lecture_01_eda/../private_notes.md",
        "lectures/lecture_01_eda/answer_keys/answers.md",
        "lectures/lecture_01_eda/solutions/walkthrough.md",
        "lectures/lecture_01_eda/grading/rubric.md",
        "lectures/lecture_01_eda/gradebook/scores.csv",
    ],
)
def test_validate_payload_rejects_non_public_concept_source_paths(
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


def test_generate_site_rejects_invalid_payload_before_writing(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    quizzes = payload["quizzes"]
    assert isinstance(quizzes, dict)
    challenge = quizzes["challenge"]
    assert isinstance(challenge, list)
    challenge.pop()
    content_path = tmp_path / "content.json"
    template_path = tmp_path / "template.html"
    output_path = tmp_path / "output" / "index.html"
    content_path.write_text(json.dumps(payload), encoding="utf-8")
    template_path.write_text("__CONTENT_JSON__", encoding="utf-8")

    with pytest.raises(ValueError, match="challenge"):
        generate_site(content_path, template_path, output_path)

    assert not output_path.exists()


def test_generate_site_writes_one_portable_html_file(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    content_path = tmp_path / "content.json"
    template_path = tmp_path / "template.html"
    output_path = tmp_path / "site" / "index.html"
    content_path.write_text(json.dumps(payload), encoding="utf-8")
    template_path.write_text(
        "<!doctype html><script>const CONTENT = __CONTENT_JSON__;</script>",
        encoding="utf-8",
    )

    result = generate_site(content_path, template_path, output_path)

    assert result == output_path
    assert output_path.is_file()
    assert not any(path.is_file() for path in output_path.parent.glob("*.*") if path != output_path)


def test_minimal_template_exposes_offline_accessibility_contract() -> None:
    template_path = (
        ROOT
        / ".agents"
        / "skills"
        / "ml-course-interactive-learning-assistant"
        / "assets"
        / "lecture-site-template.html"
    )

    template = template_path.read_text(encoding="utf-8")

    assert template.count("__CONTENT_JSON__") == 1
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in template
    assert '<main id="main-content"></main>' in template
    assert '<noscript><section id="static-content"></section></noscript>' in template
    assert "<script>const CONTENT = __CONTENT_JSON__;</script>" in template
    assert ":focus-visible" in template
    assert "prefers-reduced-motion" in template


def _valid_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
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
    <section class="graph-fallback">The same lesson is available as text.</section>
  </main>
  <noscript><section id="static-content">Static learning content.</section></noscript>
</body>
</html>
"""


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
