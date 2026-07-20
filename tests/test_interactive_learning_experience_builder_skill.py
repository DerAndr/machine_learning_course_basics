import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / ".agents" / "skills" / "interactive-learning-experience-builder"
QUIZ_STATE_MACHINE = CORE / "assets" / "quiz-state-machine.js"
QUIZ_STATE_MACHINE_TEST = ROOT / "tests" / "quiz_state_machine.test.js"


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


def test_core_skill_requires_semantic_visualizations_and_perceptible_palettes() -> None:
    skill_text = (CORE / "SKILL.md").read_text(encoding="utf-8")
    contract_text = (CORE / "references" / "content-contract.md").read_text(
        encoding="utf-8"
    )

    for phrase in (
        "topic-relevant interpretation",
        "Do not discard semantic payload fields",
        "meaningful axis, series, scenario, and control labels",
        "visibly different graph marks",
        "binary-threshold",
        "labeled-scatter",
        "residual-diagnostics",
        "coefficient-path",
        "error-metrics",
    ):
        assert phrase in skill_text or phrase in contract_text


def test_content_contract_lists_all_supported_visualization_selection_shapes() -> None:
    contract_text = (CORE / "references" / "content-contract.md").read_text(
        encoding="utf-8"
    )

    for phrase in (
        "| Type | Learning purpose | Required data shape | Control shape |",
        "`histogram`",
        "`boxplot`",
        "`scatter`",
        "`missingness`",
        "`binary-threshold`",
        "`labeled-scatter`",
        "`residual-diagnostics`",
        "`coefficient-path`",
        "`error-metrics`",
        "## Semantic visualization validation boundaries",
    ):
        assert phrase in contract_text


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
    template = (CORE / "assets" / "learning-experience-template.html").read_text(encoding="utf-8")

    assert "CONTENT.meta.experience_id" in template
    assert "CONTENT.meta.lecture_slug" not in template


def test_core_template_embeds_the_executable_quiz_state_machine() -> None:
    template = (CORE / "assets" / "learning-experience-template.html").read_text(encoding="utf-8")

    assert QUIZ_STATE_MACHINE.is_file()
    assert template.count("__QUIZ_STATE_MACHINE__") == 1


def test_quiz_state_machine_behavior_with_node() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; CI installs Node and executes this behavioral test")

    completed = subprocess.run(
        [node, "--test", str(QUIZ_STATE_MACHINE_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


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
