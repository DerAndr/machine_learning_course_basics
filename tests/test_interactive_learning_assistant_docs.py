from pathlib import Path

LIVE_URL = "https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/"
OFFLINE_PATH = "lecture_experiences/lecture_01_eda/index.html"
SKILL_PATH = ".agents/skills/ml-course-interactive-learning-assistant/SKILL.md"
CORE_SKILL_PATH = ".agents/skills/interactive-learning-experience-builder/SKILL.md"
ARCHITECTURE_PATH = "docs/learning-companions-architecture.md"
ARCHITECTURE_LINKS = {
    "README.md": "[Learning companions architecture](docs/learning-companions-architecture.md)",
    "AGENTS.md": "[Learning companions architecture](docs/learning-companions-architecture.md)",
    "docs/interactive-lecture-learning-assistant.md": (
        "[Learning Companions Architecture](learning-companions-architecture.md)"
    ),
    "docs/contributing-to-textbook.md": (
        "[Learning Companions Architecture](learning-companions-architecture.md)"
    ),
}


def test_learning_assistant_documentation_is_discoverable() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/interactive-lecture-learning-assistant.md")
    lecture = Path("lectures/lecture_01_eda/README.md").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert guide.is_file()
    guide_text = guide.read_text(encoding="utf-8")
    assert LIVE_URL in readme
    assert LIVE_URL in guide_text
    assert OFFLINE_PATH in readme
    assert OFFLINE_PATH in guide_text
    assert OFFLINE_PATH in lecture
    assert SKILL_PATH in guide_text
    assert CORE_SKILL_PATH in guide_text
    assert "context profile" in guide_text
    assert "whole-quiz Retry" in guide_text
    assert "lecture_experiences/content/" in agents
    assert "validate_learning_experience.py" in agents


def test_learning_companions_architecture_contract() -> None:
    path = Path(ARCHITECTURE_PATH)
    assert path.is_file()

    text = path.read_text(encoding="utf-8")
    for heading in (
        "# Learning Companions Architecture",
        "## What a learning companion is",
        "## Architectural layers",
        "## Responsibility boundaries",
        "## Portability model",
        "## ML-course mapping",
        "## How to use the architecture",
        "## Assurance and safety",
        "## Maintenance rules",
    ):
        assert heading in text

    for term in (
        "interactive-learning-experience-builder",
        "repository adapter",
        "experience specification",
        "grounded JSON payload",
        "deterministic",
        "self-contained",
        "file://",
        "validation",
        "student repository",
        "teacher repository",
    ):
        assert term in text

    assert "```mermaid" in text
    assert "flowchart LR" in text
    assert "| Portable core skill |" in text
    for statement in (
        "Do not create one skill per lecture or topic.",
        "The portable core remains domain-neutral.",
        "A repository adapter owns only stable local rules.",
        "Private sources are excluded at context discovery.",
    ):
        assert statement in text


def test_learning_companions_architecture_is_linked_from_repository_guides() -> None:
    for document_path, link in ARCHITECTURE_LINKS.items():
        document = Path(document_path)
        assert link in document.read_text(encoding="utf-8")

        target = link.removesuffix(")").rsplit("(", maxsplit=1)[1]
        assert (document.parent / target).is_file()

    guide_text = Path("docs/interactive-lecture-learning-assistant.md").read_text(encoding="utf-8")
    contributor_text = Path("docs/contributing-to-textbook.md").read_text(encoding="utf-8")
    assert "operational guide" in guide_text.lower()
    assert "complement" in contributor_text.lower()


def test_pages_deployment_is_limited_to_student_repository() -> None:
    workflow = Path(".github/workflows/build-textbook-preview.yml").read_text(encoding="utf-8")

    assert "github.repository == 'DerAndr/machine_learning_course_basics'" in workflow


def test_textbook_skill_requires_mobile_quiz_contract() -> None:
    text = Path(".agents/skills/ml-course-textbook-contributor/SKILL.md").read_text()

    assert "wrong answers" in text.lower()
    assert "sticky progress" in text.lower()
    assert "mobile chrome" in text.lower()
