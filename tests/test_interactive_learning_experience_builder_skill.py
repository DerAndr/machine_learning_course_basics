import importlib.util
from pathlib import Path
from types import ModuleType

import yaml


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / ".agents" / "skills" / "interactive-learning-experience-builder"


def _load_generator() -> ModuleType:
    script = CORE / "scripts" / "generate_learning_experience.py"
    spec = importlib.util.spec_from_file_location("generate_learning_experience", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_generator()
validate_payload = generator.validate_payload


def valid_payload() -> dict[str, object]:
    question = {
        "id": "foundations-1",
        "type": "single-choice",
        "prompt": "Which form is durable?",
        "options": ["Arch", "Wall"],
        "answer": "Arch",
        "explanation": "The source explains the arch.",
        "concept": "arch",
    }
    return {
        "meta": {
            "experience_id": "architecture-basics",
            "title": "Architecture Basics",
            "sources": ["knowledge/architecture.md"],
        },
        "defaults": {
            "difficulty": "foundations",
            "focus_mode": True,
            "color_blind": True,
            "break_prompts": False,
        },
        "concepts": [
            {
                "id": "arch",
                "title": "Arch",
                "explanation": "An arch transfers load to its supports.",
                "interpretation": "Read it as a force path.",
                "common_mistakes": ["Ignoring the supports."],
                "sources": ["knowledge/architecture.md"],
            }
        ],
        "visualizations": [
            {
                "id": "arch-spans",
                "type": "histogram",
                "title": "Arch spans",
                "explanation": "The distribution shows typical spans.",
                "data": [2, 3, 3, 4],
                "controls": {"bins": [2, 4]},
                "fallback": "Most spans are between 2 and 4 metres.",
            }
        ],
        "quizzes": {
            level: [{**question, "id": f"{level}-{number}"} for number in range(1, 11)]
            for level in ("foundations", "applied", "challenge")
        },
        "break_prompts": ["Take a short walk around the colonnade."],
    }


def test_core_skill_is_domain_neutral() -> None:
    text = (CORE / "SKILL.md").read_text(encoding="utf-8")

    for forbidden in ("lectures/index.yaml", "okf/", "uv run", "ML course"):
        assert forbidden not in text
    assert "context" in text.lower()
    assert "adapter" in text.lower()


def test_core_skill_contract_files_are_present() -> None:
    for path in (
        CORE / "SKILL.md",
        CORE / "agents" / "openai.yaml",
        CORE / "references" / "context-discovery.md",
        CORE / "references" / "content-contract.md",
        CORE / "references" / "repository-adapter-template.md",
    ):
        assert path.is_file()

    frontmatter = yaml.safe_load((CORE / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1])
    assert frontmatter["name"] == "interactive-learning-experience-builder"
    assert frontmatter["description"].startswith("Use when")


def test_core_template_uses_portable_experience_metadata() -> None:
    template = (CORE / "assets" / "learning-experience-template.html").read_text(
        encoding="utf-8"
    )

    assert "CONTENT.meta.experience_id" in template
    assert "CONTENT.meta.lecture_slug" not in template


def test_payload_accepts_non_ml_source_identifiers(tmp_path: Path) -> None:
    payload = valid_payload()
    meta = payload["meta"]
    assert isinstance(meta, dict)
    meta["experience_id"] = "roman-architecture"
    meta["sources"] = [
        "knowledge/architecture.md",
        "https://example.edu/reference",
        "kb:history/roman-architecture",
    ]
    concepts = payload["concepts"]
    assert isinstance(concepts, list)
    concept = concepts[0]
    assert isinstance(concept, dict)
    concept["sources"] = meta["sources"]
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "architecture.md").write_text("Source", encoding="utf-8")

    assert validate_payload(payload, repository_root=tmp_path) == []
