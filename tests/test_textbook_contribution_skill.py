from __future__ import annotations

from pathlib import Path

import yaml


def test_textbook_contributor_skill_is_complete() -> None:
    skill_dir = Path(".agents/skills/ml-course-textbook-contributor")
    skill = skill_dir / "SKILL.md"
    reference = skill_dir / "references" / "contribution-workflow.md"

    assert skill.is_file()
    assert reference.is_file()
    for agent_file in ["openai.yaml", "gemini.yaml", "claude.yaml"]:
        assert (skill_dir / "agents" / agent_file).is_file()

    text = skill.read_text(encoding="utf-8")
    assert "TODO" not in text
    assert "ml-course-textbook-contributor" in text
    assert "uv run python tools/validate_okf.py okf/ --strict-warnings" in text

    frontmatter = text.split("---", 2)[1]
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == "ml-course-textbook-contributor"
    assert "interactive textbook" in metadata["description"]


def test_contribution_guide_mentions_agent_skill() -> None:
    guide = Path("docs/contributing-to-textbook.md").read_text(encoding="utf-8")
    assert ".agents/skills/ml-course-textbook-contributor/SKILL.md" in guide
    assert "skills are generated from `learning_objectives`" in guide
