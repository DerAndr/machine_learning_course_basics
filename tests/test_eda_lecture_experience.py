import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / "lecture_experiences" / "content" / "lecture_01_eda.json"
SITE_PATH = ROOT / "lecture_experiences" / "lecture_01_eda" / "index.html"
TEMPLATE_PATH = (
    ROOT
    / ".agents"
    / "skills"
    / "interactive-learning-experience-builder"
    / "assets"
    / "learning-experience-template.html"
)
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
        / "interactive-learning-experience-builder"
        / "scripts"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _okf_hashes() -> dict[Path, str]:
    return {
        path.relative_to(ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "okf").rglob("*"))
        if path.is_file()
    }


def _outlier_count(values: list[float], multiplier: float) -> int:
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    q1 = quantile(0.25)
    q3 = quantile(0.75)
    iqr = q3 - q1
    return sum(value < q1 - multiplier * iqr or value > q3 + multiplier * iqr for value in ordered)


def test_eda_payload_and_generated_site_meet_learning_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))

    assert payload["meta"]["experience_id"] == "lecture-01-eda"
    assert {concept["id"] for concept in payload["concepts"]} == EXPECTED_CONCEPTS
    assert {item["type"] for item in payload["visualizations"]} == {
        "histogram",
        "boxplot",
        "scatter",
        "missingness",
    }
    assert all(len(payload["quizzes"][level]) == 10 for level in LEVELS)
    questions = [question for level in LEVELS for question in payload["quizzes"][level]]
    assert len({question["id"] for question in questions}) == 30
    assert all(question["answer"].strip() for question in questions)
    assert all(question["answer"] in question["options"] for question in questions)
    assert all(question["explanation"].strip() for question in questions)
    assert all(question["concept"] in EXPECTED_CONCEPTS for question in questions)
    assert all("lectures/lecture_01_eda/" in source for source in payload["meta"]["sources"])

    boxplot = next(item for item in payload["visualizations"] if item["type"] == "boxplot")
    outlier_counts = {
        _outlier_count(boxplot["data"], multiplier)
        for multiplier in boxplot["controls"]["fence_multipliers"]
    }
    assert len(outlier_counts) > 1

    html = SITE_PATH.read_text(encoding="utf-8")
    assert "__CONTENT_JSON__" not in html
    assert "__STATIC_CONTENT__" not in html
    assert payload["meta"]["title"] in html
    assert _load_script("validate_learning_experience").validate_html(SITE_PATH) == []

    okf_before = _okf_hashes()
    generated_path = _load_script("generate_learning_experience").write_site(
        CONTENT_PATH,
        TEMPLATE_PATH,
        tmp_path / "index.html",
        repository_root=ROOT,
    )
    assert generated_path.read_bytes() == SITE_PATH.read_bytes()
    assert _okf_hashes() == okf_before
