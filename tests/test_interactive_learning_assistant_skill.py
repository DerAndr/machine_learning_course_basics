from pathlib import Path

import yaml

SKILL_DIR = Path(".agents/skills/ml-course-interactive-learning-assistant")


def test_interactive_learning_assistant_skill_contract() -> None:
    skill = SKILL_DIR / "SKILL.md"
    metadata_file = SKILL_DIR / "agents/openai.yaml"

    assert skill.is_file()
    assert metadata_file.is_file()

    text = skill.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["name"] == "ml-course-interactive-learning-assistant"
    assert frontmatter["description"].startswith("Use when")
    for phrase in (
        "lecture_notes.md",
        "exactly 10",
        "file://",
        "color-blind",
        "focus-friendly",
        "Do not modify `okf/`",
        "validate_lecture_site.py",
    ):
        assert phrase in text

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert metadata["interface"]["display_name"] == ("Interactive Lecture Learning Assistant")
