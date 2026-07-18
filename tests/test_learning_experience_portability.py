import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / ".agents" / "skills" / "interactive-learning-experience-builder"


def _history_payload() -> dict[str, object]:
    question = {
        "type": "single-choice",
        "prompt": "Why compare primary accounts?",
        "options": ["They can reveal different perspectives", "They eliminate interpretation"],
        "answer": "They can reveal different perspectives",
        "explanation": "Different accounts can describe the same event from distinct positions.",
        "concept": "source-comparison",
    }
    return {
        "meta": {
            "experience_id": "history-source-comparison",
            "title": "Comparing Historical Sources",
            "sources": ["knowledge/history.md"],
        },
        "defaults": {
            "difficulty": "foundations",
            "focus_mode": False,
            "color_blind": True,
            "break_prompts": False,
        },
        "concepts": [
            {
                "id": "source-comparison",
                "title": "Compare perspectives",
                "explanation": (
                    "Historical accounts reflect their authors, audiences, and purposes."
                ),
                "interpretation": (
                    "Compare claims with the context in which each source was created."
                ),
                "common_mistakes": ["Treating one source as a complete account."],
                "sources": ["knowledge/history.md"],
            }
        ],
        "visualizations": [
            {
                "id": "account-counts",
                "type": "histogram",
                "title": "Accounts by year",
                "explanation": "The bars show how many accounts survive from each year.",
                "data": [1, 2, 2, 3],
                "controls": {"bins": [2, 4]},
                "fallback": "Most surviving accounts fall in the middle two year groups.",
            }
        ],
        "quizzes": {
            level: [{**question, "id": f"{level}-{number}"} for number in range(1, 11)]
            for level in ("foundations", "applied", "challenge")
        },
        "break_prompts": ["Pause and compare two viewpoints before continuing."],
    }


def test_core_generates_in_an_unrelated_repository_without_course_tooling(
    tmp_path: Path,
) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "history.md").write_text(
        "Compare primary accounts with their context.", encoding="utf-8"
    )
    (tmp_path / "content.json").write_text(json.dumps(_history_payload()), encoding="utf-8")
    shutil.copytree(CORE, tmp_path / ".agents" / "skills" / CORE.name)

    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "uv.lock").exists()
    assert not (tmp_path / "publish").exists()

    copied_core = tmp_path / ".agents" / "skills" / CORE.name
    generator = copied_core / "scripts" / "generate_learning_experience.py"
    validator = copied_core / "scripts" / "validate_learning_experience.py"
    template = copied_core / "assets" / "learning-experience-template.html"
    output = tmp_path / "site" / "index.html"

    generated = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--content",
            "content.json",
            "--template",
            str(template.relative_to(tmp_path)),
            "--output",
            "site/index.html",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert output.is_file()

    validated = subprocess.run(
        [sys.executable, str(validator), "site/index.html"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert validated.returncode == 0, validated.stdout + validated.stderr
    assert "VALID:" in validated.stdout
