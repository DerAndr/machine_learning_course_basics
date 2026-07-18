from pathlib import Path

import yaml

SKILL_DIR = Path(".agents/skills/ml-course-interactive-learning-assistant")


def test_interactive_learning_assistant_skill_contract() -> None:
    skill = SKILL_DIR / "SKILL.md"
    metadata_file = SKILL_DIR / "agents/openai.yaml"
    content_contract_file = SKILL_DIR / "references/content-contract.md"

    assert skill.is_file()
    assert metadata_file.is_file()
    assert content_contract_file.is_file()

    text = skill.read_text(encoding="utf-8")
    content_contract = content_contract_file.read_text(encoding="utf-8")
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

    static_fallback = (
        "Keep explanations and full quiz content statically readable if JavaScript fails."
    )
    assert static_fallback in text
    assert static_fallback in content_contract
    assert "Always embed `break_prompts` content" in content_contract
    assert "controls only the initial state" in content_contract
    assert "labels, shapes, patterns, or line styles in addition to color" in content_contract

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert metadata["interface"]["display_name"] == ("Interactive Lecture Learning Assistant")
