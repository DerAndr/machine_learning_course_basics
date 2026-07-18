import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / "lecture_experiences" / "content" / "lecture_01_eda.json"
SITE_PATH = ROOT / "lecture_experiences" / "lecture_01_eda" / "index.html"
LEVELS = ("foundations", "applied", "challenge")
EXPECTED_CONCEPTS = {
    "eda-before-modeling",
    "data-types-and-structure",
    "center-spread-skew",
    "histogram-bins",
    "iqr-boxplot-outliers",
    "scatter-association",
    "missing-counts-proportions",
    "automation-with-reasoning",
}


def _load_script(name: str) -> ModuleType:
    path = (
        ROOT
        / ".agents"
        / "skills"
        / "ml-course-interactive-learning-assistant"
        / "scripts"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eda_payload_and_generated_site_meet_learning_contract() -> None:
    payload = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))

    assert payload["meta"]["lecture_slug"] == "lecture_01_eda"
    assert {concept["id"] for concept in payload["concepts"]} == EXPECTED_CONCEPTS
    assert {item["type"] for item in payload["visualizations"]} == {
        "histogram",
        "boxplot",
        "scatter",
        "missingness",
    }
    assert all(len(payload["quizzes"][level]) == 10 for level in LEVELS)
    questions = [
        question for level in LEVELS for question in payload["quizzes"][level]
    ]
    assert len({question["id"] for question in questions}) == 30
    assert all(question["explanation"].strip() for question in questions)
    assert all(question["concept"] in EXPECTED_CONCEPTS for question in questions)
    assert all(
        "lectures/lecture_01_eda/" in source
        for source in payload["meta"]["sources"]
    )

    html = SITE_PATH.read_text(encoding="utf-8")
    assert "__CONTENT_JSON__" not in html
    assert "__STATIC_CONTENT__" not in html
    assert payload["meta"]["title"] in html
    assert _load_script("validate_lecture_site").validate_html(SITE_PATH) == []

    tracked_changes = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    untracked_changes = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    changed_paths = tracked_changes + untracked_changes
    assert all("okf/" not in changed_path for changed_path in changed_paths)
