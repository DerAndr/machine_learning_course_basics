import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = (
    ROOT
    / "lecture_experiences"
    / "content"
    / "lecture_05_classification_part_1.json"
)
SITE_PATH = (
    ROOT
    / "lecture_experiences"
    / "lecture_05_classification_part_1"
    / "index.html"
)
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
LECTURE_SLUG = "lecture_05_classification_part_1"
EXPERIENCE_ID = "lecture-05-classification-part-1"
LEVELS = ("foundations", "applied", "challenge")
EXPECTED_CONCEPTS = {
    "classification-problem-types",
    "knn-distance-scaling",
    "decision-tree-impurity",
    "tree-overfitting-control",
    "logistic-probabilities-thresholds",
    "confusion-matrix-outcomes",
    "precision-recall-fscore",
    "roc-auc-log-loss",
}
EXPECTED_VISUALIZATION_TYPES = {"binary-threshold", "labeled-scatter"}
EXPECTED_VISUALIZATIONS = {
    "cls-threshold-confusion": "binary-threshold",
    "cls-decision-boundary": "labeled-scatter",
}
EXPECTED_DEFAULTS = {
    "difficulty": "foundations",
    "focus_mode": True,
    "color_blind": True,
    "break_prompts": True,
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


def _source_is_allowed(source: str) -> bool:
    return source.startswith(f"lectures/{LECTURE_SLUG}/") or source.startswith(
        "okf/"
    )


def test_classification_payload_and_generated_site_meet_learning_contract(
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
    assert payload["defaults"] == EXPECTED_DEFAULTS
    assert {concept["id"] for concept in payload["concepts"]} == EXPECTED_CONCEPTS
    assert {item["type"] for item in payload["visualizations"]} == (
        EXPECTED_VISUALIZATION_TYPES
    )
    assert {
        item["id"]: item["type"] for item in payload["visualizations"]
    } == EXPECTED_VISUALIZATIONS
    assert all(len(payload["quizzes"][level]) == 10 for level in LEVELS)

    questions = [
        question for level in LEVELS for question in payload["quizzes"][level]
    ]
    expected_question_ids = {
        f"cls-{prefix}-{number:02d}"
        for prefix in ("f", "a", "c")
        for number in range(1, 11)
    }
    assert {question["id"] for question in questions} == expected_question_ids
    assert len({question["id"] for question in questions}) == 30
    assert all(question["answer"] for question in questions)
    assert all(question["explanation"].strip() for question in questions)
    assert all(question["concept"] in EXPECTED_CONCEPTS for question in questions)

    for question in questions:
        if question["type"] == "single-choice":
            assert question["answer"] in question["options"]
        elif question["type"] == "multiple-choice":
            assert isinstance(question["answer"], list)
            assert len(question["answer"]) == len(set(question["answer"]))
            assert set(question["answer"]) <= set(question["options"])

    assert all(_source_is_allowed(source) for source in payload["meta"]["sources"])
    assert all(
        _source_is_allowed(source)
        for concept in payload["concepts"]
        for source in concept["sources"]
    )

    visualizations = payload["visualizations"]
    assert [visualization["type"] for visualization in visualizations] == [
        "binary-threshold",
        "labeled-scatter",
    ]
    assert {
        item["id"]: item["type"] for item in visualizations
    } == EXPECTED_VISUALIZATIONS

    threshold = visualizations[0]
    assert threshold["controls"]["initial"] == 0.5
    assert {record["actual"] for record in threshold["data"]} == {0, 1}

    boundary = visualizations[1]
    assert {point["series"] for point in boundary["data"]} == {"A", "B"}
    assert boundary["labels"]["positive_series"] == "B"
    assert len(boundary["controls"]["boundaries"]) == 3

    html = SITE_PATH.read_text(encoding="utf-8")
    assert "__CONTENT_JSON__" not in html
    assert "__STATIC_CONTENT__" not in html
    assert "__QUIZ_STATE_MACHINE__" not in html
    assert "LearningExperienceQuiz" in html
    assert payload["meta"]["title"] in html
    for hook in (
        "Decision threshold",
        "confusion matrix",
        "Precision",
        "recall",
        "Balanced boundary",
        "Conservative positive boundary",
        "Permissive positive boundary",
    ):
        assert hook in html
    assert _load_script("validate_learning_experience").validate_html(SITE_PATH) == []

    okf_before = _okf_hashes()
    course_generator_spec = importlib.util.spec_from_file_location(
        "generate_course_learning_experience",
        COURSE_GENERATOR_PATH,
    )
    assert course_generator_spec is not None
    assert course_generator_spec.loader is not None
    course_generator = importlib.util.module_from_spec(course_generator_spec)
    course_generator_spec.loader.exec_module(course_generator)
    generated_path = course_generator.generate_course_site(
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
