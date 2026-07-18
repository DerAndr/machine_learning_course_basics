from pathlib import Path

import yaml

SKILL_DIR = Path(".agents/skills/ml-course-interactive-learning-assistant")
SKILL_PATH = SKILL_DIR / "SKILL.md"


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
        "generate_learning_experience.py",
        "learning-experience-template.html",
        "validate_learning_experience.py",
    ):
        assert phrase in text

    assert "ml-course-interactive-learning-assistant/scripts" not in text

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert metadata["interface"]["display_name"] == ("Interactive Lecture Learning Assistant")


def test_adapter_preserves_course_constraints() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    for required in (
        "lectures/index.yaml",
        "lecture_notes.md",
        "okf/",
        "Do not modify `okf/`",
        "exactly 10",
        "interactive-learning-experience-builder",
    ):
        assert required in text
