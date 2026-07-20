import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(".agents/skills/ml-course-interactive-learning-assistant")
SKILL_PATH = SKILL_DIR / "SKILL.md"
POLICY_WRAPPER = SKILL_DIR / "scripts/generate_course_learning_experience.py"
CORE_TEMPLATE = (
    Path(".agents/skills/interactive-learning-experience-builder")
    / "assets"
    / "learning-experience-template.html"
)
REAL_TEACHER_SOURCES = (
    "lectures/lecture_01_eda/practical_session/teacher_cheat_sheet.md",
    "lectures/lecture_01_eda/practical_session/eda_practical_teacher_90min.ipynb",
)


def test_interactive_learning_assistant_skill_contract() -> None:
    metadata_file = SKILL_DIR / "agents/openai.yaml"

    assert SKILL_PATH.is_file()
    assert metadata_file.is_file()
    assert not (SKILL_DIR / "references/content-contract.md").exists()

    text = SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["name"] == "ml-course-interactive-learning-assistant"
    assert frontmatter["description"].startswith("Use when")
    for phrase in (
        "file://",
        "color-blind",
        "focus-friendly",
        "generate_course_learning_experience.py",
        "learning-experience-template.html",
        "validate_learning_experience.py",
    ):
        assert phrase in text

    assert POLICY_WRAPPER.is_file()
    assert not (SKILL_DIR / "assets/learning-experience-template.html").exists()
    assert not (SKILL_DIR / "scripts/generate_learning_experience.py").exists()

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert metadata["interface"]["display_name"] == ("Interactive Lecture Learning Assistant")


def test_adapter_preserves_course_constraints() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    for required in (
        "lectures/index.yaml",
        "lecture_notes.md",
        "okf/",
        "Do not modify `okf/`",
        "interactive-learning-experience-builder",
    ):
        assert required in text


def test_adapter_requires_objective_matched_semantic_visualizations() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    for phrase in (
        "named lecture objective",
        "repeated generic chart set",
        "semantic visualization type",
        "exercise every visualization control",
        "both palette modes",
    ):
        assert phrase in text


def test_adapter_preserves_learning_experience_parity() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    for required in (
        "Foundations",
        "Applied",
        "Challenge",
        "focus-friendly mode",
        "color-blind-safe palette",
        "funny topic-related break prompts",
        "`foundations`, `applied`, and `challenge` quiz banks",
        "exactly 10 questions each",
        "static no-JavaScript explanations and quiz review",
        "deterministic single-file output",
        "accessible chart fallbacks",
        "keyboard navigation",
        "reduced motion",
        "storage fallback",
        "answer review",
        "whole-quiz Retry",
        "generate_course_learning_experience.py",
        "validate_learning_experience.py",
        "private solutions",
        "teacher notebooks",
        "answer keys",
        "grading data",
    ):
        assert required in normalized_text


def _load_policy_wrapper():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_course_learning_experience",
        POLICY_WRAPPER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _materialize_repository_source(repository_root: Path, source: str) -> Path | None:
    if ":" in source:
        return None
    path = repository_root.joinpath(*source.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Synthetic policy fixture.", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "source",
    [
        "lectures/lecture_01_eda/private/notes.md",
        "lectures/lecture_01_eda/solutions/walkthrough.md",
        "lectures/lecture_01_eda/quizzes/questions.json",
        "lectures/lecture_01_eda/grading/rubric.md",
        *REAL_TEACHER_SOURCES,
    ],
)
def test_course_policy_behaviorally_rejects_restricted_sources(
    tmp_path: Path,
    source: str,
) -> None:
    fixture = _materialize_repository_source(tmp_path, source)
    assert fixture is not None
    assert fixture.is_file()
    validator = _load_policy_wrapper()
    payload = {"meta": {"sources": [source]}, "concepts": []}

    errors = validator.validate_course_source_policy(payload, "lecture_01_eda")

    assert errors
    assert any("restricted" in error for error in errors)


def test_course_policy_rejects_a_different_lecture_source(tmp_path: Path) -> None:
    validator = _load_policy_wrapper()
    source = "lectures/lecture_02_data_preparation_part_1/lecture_notes.md"
    fixture = _materialize_repository_source(tmp_path, source)
    assert fixture is not None
    assert fixture.is_file()
    payload = {
        "meta": {"sources": [source]},
        "concepts": [],
    }

    errors = validator.validate_course_source_policy(payload, "lecture_01_eda")

    assert errors == [
        "source belongs to a different lecture than lecture_01_eda: "
        "lectures/lecture_02_data_preparation_part_1/lecture_notes.md"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "docs/interactive-lecture-learning-assistant.md",
        "quizzes/README.md",
        "https://example.edu/reference",
        "kb:history/roman-architecture",
        "Lectures/lecture_01_eda/lecture_notes.md",
        "OKF/course-overview/course-overview.md",
    ],
)
def test_course_policy_rejects_every_source_outside_canonical_roots(
    tmp_path: Path,
    source: str,
) -> None:
    fixture = _materialize_repository_source(tmp_path, source)
    if fixture is not None:
        assert fixture.is_file()
    validator = _load_policy_wrapper()
    payload = {"meta": {"sources": [source]}, "concepts": []}

    errors = validator.validate_course_source_policy(payload, "lecture_01_eda")

    assert errors
    assert any(source in error for error in errors)


def test_course_policy_allows_selected_lecture_and_read_only_okf_sources() -> None:
    validator = _load_policy_wrapper()
    payload = {
        "meta": {
            "sources": [
                "lectures/lecture_01_eda/lecture_notes.md",
                "okf/course-overview/course-overview.md",
            ]
        },
        "concepts": [
            {
                "sources": [
                    "lectures/lecture_01_eda/practical_session/README.md",
                    "okf/labs/classification-threshold-explorer.md",
                ]
            }
        ],
    }

    assert validator.validate_course_source_policy(payload, "lecture_01_eda") == []


@pytest.mark.parametrize(
    "source",
    [
        *REAL_TEACHER_SOURCES,
        "quizzes/README.md",
        "https://example.edu/reference",
        "kb:history/roman-architecture",
        "Lectures/lecture_01_eda/lecture_notes.md",
        "OKF/course-overview/course-overview.md",
    ],
)
def test_course_policy_wrapper_cli_rejects_before_writing_output(
    tmp_path: Path,
    source: str,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    fixture = _materialize_repository_source(repository_root, source)
    if fixture is not None:
        assert fixture.is_file()
    content = repository_root / "content.json"
    output = repository_root / "index.html"
    content.write_text(
        json.dumps(
            {
                "meta": {
                    "sources": [source],
                },
                "concepts": [],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(POLICY_WRAPPER.resolve()),
            "--lecture-slug",
            "lecture_01_eda",
            "--content",
            str(content),
            "--template",
            str(CORE_TEMPLATE.resolve()),
            "--output",
            str(output),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert source in completed.stdout
    assert not output.exists()


@pytest.mark.parametrize(
    "workflow_path",
    [
        Path(".github/workflows/build-textbook-preview.yml"),
        Path(".github/workflows/validate-okf.yml"),
    ],
)
def test_workflows_cover_portable_scripts_and_tests(workflow_path: Path) -> None:
    workflow = workflow_path.read_text(encoding="utf-8")

    for required in (
        ".agents/skills/interactive-learning-experience-builder/scripts/"
        "generate_learning_experience.py",
        ".agents/skills/interactive-learning-experience-builder/scripts/"
        "validate_learning_experience.py",
        ".agents/skills/ml-course-interactive-learning-assistant/scripts/"
        "generate_course_learning_experience.py",
        "tests/test_interactive_learning_experience_builder_skill.py",
        "tests/test_learning_experience_portability.py",
    ):
        assert required in workflow
