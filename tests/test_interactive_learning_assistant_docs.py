from pathlib import Path

DEMOS = {
    "lecture_01_eda": {
        "live": ("https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/"),
        "offline": "lecture_experiences/lecture_01_eda/index.html",
        "lecture_readme": "lectures/lecture_01_eda/README.md",
    },
    "lecture_04_regression": {
        "live": (
            "https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/"
        ),
        "offline": "lecture_experiences/lecture_04_regression/index.html",
        "lecture_readme": "lectures/lecture_04_regression/README.md",
    },
    "lecture_05_classification_part_1": {
        "live": (
            "https://derandr.github.io/machine_learning_course_basics/"
            "demos/lecture_05_classification_part_1/"
        ),
        "offline": ("lecture_experiences/lecture_05_classification_part_1/index.html"),
        "lecture_readme": ("lectures/lecture_05_classification_part_1/README.md"),
    },
}
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
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert guide.is_file()
    guide_text = guide.read_text(encoding="utf-8")
    for demo in DEMOS.values():
        assert demo["live"] in readme
        assert demo["offline"] in readme
        assert demo["live"] in guide_text
        assert demo["offline"] in guide_text
        lecture_text = Path(demo["lecture_readme"]).read_text(encoding="utf-8")
        assert demo["live"] in lecture_text
        assert demo["offline"] in lecture_text

    for heading in (
        "## Choose how to learn",
        "## Fast interactive reviews",
        "## Course map",
        "## Create interactive learning materials",
        "## Local setup",
        "## Contributing and reference",
        "## Repository map",
        "## License",
    ):
        assert heading in readme

    general_prompt = (
        "Use $interactive-learning-experience-builder to create a grounded, "
        "offline interactive learning experience from this repository's "
        "knowledge sources."
    )
    course_prompt = (
        "Use $ml-course-interactive-learning-assistant with "
        "$interactive-learning-experience-builder to create a grounded, "
        "accessible, self-contained review for a selected ML-course lecture."
    )
    for heading in (
        "## Choose the right skill",
        "## General-purpose workflow",
        "## ML-course workflow",
        "## Authoring and verification flow",
    ):
        assert heading in guide_text
    assert general_prompt in guide_text
    assert course_prompt in guide_text

    okf_index = Path("okf/index.md").read_text(encoding="utf-8")
    assert "## Fast interactive reviews" in okf_index
    assert "complement" in okf_index.lower()
    assert "demos/" not in okf_index
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
