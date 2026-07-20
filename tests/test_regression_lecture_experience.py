import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / "lecture_experiences" / "content" / "lecture_04_regression.json"
SITE_PATH = ROOT / "lecture_experiences" / "lecture_04_regression" / "index.html"
TEMPLATE_PATH = (
    ROOT
    / ".agents"
    / "skills"
    / "interactive-learning-experience-builder"
    / "assets"
    / "learning-experience-template.html"
)
COURSE_GENERATOR_PATH = (
    ROOT
    / ".agents"
    / "skills"
    / "ml-course-interactive-learning-assistant"
    / "scripts"
    / "generate_course_learning_experience.py"
)
LECTURE_SLUG = "lecture_04_regression"
EXPERIENCE_ID = "lecture-04-regression"
LEVELS = ("foundations", "applied", "challenge")
EXPECTED_CONCEPTS = {
    "regression-problem-types",
    "ols-fitted-values-residuals",
    "assumptions-as-diagnostics",
    "multicollinearity-instability",
    "ridge-lasso-regularization",
    "scaling-for-regularization",
    "regression-metrics",
    "validation-overfitting-interpretation",
}
EXPECTED_VISUALIZATION_TYPES = {
    "residual-diagnostics",
    "coefficient-path",
    "error-metrics",
}


def _load_core_script(name: str) -> ModuleType:
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


def _load_course_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_course_learning_experience",
        COURSE_GENERATOR_PATH,
    )
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


def test_regression_payload_and_generated_site_meet_learning_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))

    assert set(payload) == {
        "meta",
        "defaults",
        "concepts",
        "visualizations",
        "quizzes",
        "break_prompts",
    }
    assert payload["meta"]["experience_id"] == EXPERIENCE_ID
    assert {concept["id"] for concept in payload["concepts"]} == EXPECTED_CONCEPTS
    assert {
        item["type"] for item in payload["visualizations"]
    } == EXPECTED_VISUALIZATION_TYPES
    visualizations = payload["visualizations"]
    assert [visualization["type"] for visualization in visualizations] == [
        "residual-diagnostics",
        "coefficient-path",
        "error-metrics",
    ]

    residuals = visualizations[0]
    assert [scenario["id"] for scenario in residuals["data"]["scenarios"]] == [
        "appropriate",
        "curvature",
        "funnel",
    ]

    paths = visualizations[1]
    assert paths["data"]["penalties"] == [0, 0.1, 0.5, 1, 2]
    assert all(series["lasso"][-1] == 0 for series in paths["data"]["series"])

    metrics = visualizations[2]
    assert metrics["data"]["adjustable_error"] == [0, 5, 10, 20]
    assert all(len(payload["quizzes"][level]) == 10 for level in LEVELS)

    questions = [
        question for level in LEVELS for question in payload["quizzes"][level]
    ]
    assert len({question["id"] for question in questions}) == 30
    assert [question["id"] for question in payload["quizzes"]["foundations"]] == [
        f"reg-f-{number:02d}" for number in range(1, 11)
    ]
    assert [question["id"] for question in payload["quizzes"]["applied"]] == [
        f"reg-a-{number:02d}" for number in range(1, 11)
    ]
    assert [question["id"] for question in payload["quizzes"]["challenge"]] == [
        f"reg-c-{number:02d}" for number in range(1, 11)
    ]
    assert all(question["answer"] for question in questions)
    assert all(question["explanation"].strip() for question in questions)
    assert all(question["concept"] in EXPECTED_CONCEPTS for question in questions)
    for question in questions:
        if question["type"] == "single-choice":
            assert question["answer"] in question["options"]
        elif question["type"] == "multiple-choice":
            assert set(question["answer"]).issubset(question["options"])
        elif question["type"] == "interpretation" and question["options"]:
            assert question["answer"] in question["options"]

    assert all(
        source.startswith("lectures/lecture_04_regression/")
        or source.startswith("okf/")
        for source in payload["meta"]["sources"]
    )
    assert all(
        source.startswith("lectures/lecture_04_regression/")
        or source.startswith("okf/")
        for concept in payload["concepts"]
        for source in concept["sources"]
    )

    assert all(item["fallback"].strip() for item in payload["visualizations"])

    html = SITE_PATH.read_text(encoding="utf-8")
    assert "__CONTENT_JSON__" not in html
    assert "__STATIC_CONTENT__" not in html
    assert "__QUIZ_STATE_MACHINE__" not in html
    assert "LearningExperienceQuiz" in html
    assert payload["meta"]["title"] in html
    for hook in (
        "Appropriate linear fit",
        "Curved residual pattern",
        "Funnel-shaped variance",
        "Ridge",
        "Lasso",
        "MAE",
        "RMSE",
    ):
        assert hook in html
    assert _load_core_script("validate_learning_experience").validate_html(SITE_PATH) == []

    okf_before = _okf_hashes()
    generated_path = _load_course_generator().generate_course_site(
        CONTENT_PATH,
        TEMPLATE_PATH,
        tmp_path / "index.html",
        LECTURE_SLUG,
        repository_root=ROOT,
    )
    generated_bytes = generated_path.read_bytes()
    committed_bytes = SITE_PATH.read_bytes()
    assert b"\r\n" not in generated_bytes
    assert b"\r\n" not in committed_bytes
    assert generated_bytes == committed_bytes
    assert _okf_hashes() == okf_before
