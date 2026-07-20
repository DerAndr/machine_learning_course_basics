import re
from pathlib import Path

DEMOS = {
    "lecture_01_eda": {
        "live": ("https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/"),
        "offline": "lecture_experiences/lecture_01_eda/index.html",
        "lecture_readme": "lectures/lecture_01_eda/README.md",
    },
    "lecture_04_regression": {
        "live": (
            "https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/"
        ),
        "offline": "lecture_experiences/lecture_04_regression/index.html",
        "lecture_readme": "lectures/lecture_04_regression/README.md",
    },
    "lecture_05_classification_part_1": {
        "live": (
            "https://derandr.github.io/machine_learning_course_basics/"
            "demos/lecture_05_classification_part_1/"
        ),
        "offline": ("lecture_experiences/lecture_05_classification_part_1/index.html"),
        "lecture_readme": ("lectures/lecture_05_classification_part_1/README.md"),
    },
}
SKILL_PATH = ".agents/skills/ml-course-interactive-learning-assistant/SKILL.md"
CORE_SKILL_PATH = ".agents/skills/interactive-learning-experience-builder/SKILL.md"
ARCHITECTURE_PATH = "docs/learning-companions-architecture.md"
SUBMISSION_PREPARATION_PATH = "docs/build-week-submission-preparation.md"
STUDENT_COMPANION_QUICKSTART_PATH = "docs/student-learning-companion-quickstart.md"
ARCHITECTURE_LINKS = {
    "README.md": "[Learning companions architecture](docs/learning-companions-architecture.md)",
    "AGENTS.md": "[Learning companions architecture](docs/learning-companions-architecture.md)",
    "docs/interactive-lecture-learning-assistant.md": (
        "[Learning Companions Architecture](learning-companions-architecture.md)"
    ),
    "docs/contributing-to-textbook.md": (
        "[Learning Companions Architecture](learning-companions-architecture.md)"
    ),
}
SEMANTIC_COMPANION_TERMS = {
    "README.md": (
        "threshold and confusion matrix",
        "decision boundary",
        "residual diagnostics",
        "Ridge and Lasso",
        "MAE and RMSE",
    ),
    "docs/learning-companions-architecture.md": (
        "visualization-models.js",
        "semantic visualization payload",
        "pure visualization model",
    ),
    "docs/interactive-lecture-learning-assistant.md": (
        "topic-relevant interpretation",
        "both palette modes",
        "exercise every visualization control",
    ),
    "docs/student-learning-companion-quickstart.md": (
        "topic-specific interaction",
        "trusted source",
        "color-blind-safe",
    ),
    "docs/build-week-integration-evidence.md": (
        "ignored class semantics",
        "perceptually weak",
        "shared portable core",
    ),
    "docs/build-week-submission-preparation.md": (
        "feedback",
        "threshold",
        "regularization",
        "metric sensitivity",
        "palette",
    ),
}


def test_semantic_companion_story_is_consistent_across_public_docs() -> None:
    for path, phrases in SEMANTIC_COMPANION_TERMS.items():
        text = Path(path).read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase.lower() in text.lower(), f"{path} must explain {phrase!r}"


def test_student_learning_companion_quickstart_is_actionable() -> None:
    path = Path(STUDENT_COMPANION_QUICKSTART_PATH)
    assert path.is_file()

    text = path.read_text(encoding="utf-8")
    for heading in (
        "# Student Learning Companion Quickstart",
        "## Pick the shortest path",
        "## Use the skills from this repository",
        "## Course-specific prompt examples",
        "## Use the generic skill in another repository",
        "## Add the generic skill to your personal Codex",
        "## Troubleshooting",
    ):
        assert heading in text

    for required_item in (
        "$interactive-learning-experience-builder",
        "$ml-course-interactive-learning-assistant",
        "$HOME/.agents/skills",
        "/skills",
        "lecture_04_regression",
        "Trusted sources",
        "Excluded material",
        "Copy-Item",
        "cp -R",
        "https://learn.chatgpt.com/docs/build-skills",
    ):
        assert required_item in text

    links = {
        "README.md": (
            "[student prompt and installation quickstart]"
            "(docs/student-learning-companion-quickstart.md)"
        ),
        "docs/student-quickstart.md": (
            "[student learning-companion quickstart](student-learning-companion-quickstart.md)"
        ),
        "docs/interactive-lecture-learning-assistant.md": (
            "[student learning-companion quickstart](student-learning-companion-quickstart.md)"
        ),
    }
    for document_path, link in links.items():
        assert link in Path(document_path).read_text(encoding="utf-8")


def test_learning_assistant_documentation_is_discoverable() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/interactive-lecture-learning-assistant.md")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert guide.is_file()
    guide_text = guide.read_text(encoding="utf-8")
    for demo in DEMOS.values():
        assert demo["live"] in readme
        assert demo["offline"] in readme
        assert demo["live"] in guide_text
        assert demo["offline"] in guide_text
        lecture_text = Path(demo["lecture_readme"]).read_text(encoding="utf-8")
        assert demo["live"] in lecture_text
        assert demo["offline"] in lecture_text

    for heading in (
        "## Choose how to learn",
        "## Fast interactive reviews",
        "## Course map",
        "## Create interactive learning materials",
        "## How Codex and GPT-5.6 were used",
        "## Local setup",
        "## Contributing and reference",
        "## Repository map",
        "## License",
    ):
        assert heading in readme
    assert "docs/build-week-submission-preparation.md" in readme
    assert "human" in readme.lower()

    general_prompt = (
        "Use $interactive-learning-experience-builder to create a grounded, "
        "offline interactive learning experience from this repository's "
        "knowledge sources."
    )
    course_prompt = (
        "Use $ml-course-interactive-learning-assistant with "
        "$interactive-learning-experience-builder to create a grounded, "
        "accessible, self-contained review for a selected ML-course lecture."
    )
    for heading in (
        "## Choose the right skill",
        "## General-purpose workflow",
        "## ML-course workflow",
        "## Authoring and verification flow",
    ):
        assert heading in guide_text
    assert general_prompt in guide_text
    assert course_prompt in guide_text

    okf_index = Path("okf/index.md").read_text(encoding="utf-8")
    assert "## Fast interactive reviews" in okf_index
    assert "complement" in okf_index.lower()
    assert "demos/" not in okf_index
    assert SKILL_PATH in guide_text
    assert CORE_SKILL_PATH in guide_text
    assert "context profile" in guide_text
    assert "whole-quiz Retry" in guide_text
    assert "lecture_experiences/content/" in agents
    assert "validate_learning_experience.py" in agents


def test_build_week_submission_preparation_is_actionable() -> None:
    path = Path(SUBMISSION_PREPARATION_PATH)
    assert path.is_file()

    text = path.read_text(encoding="utf-8")
    for heading in (
        "# Build Week Submission Preparation",
        "## Narrative spine",
        "## Devpost field worksheet",
        "## Judge test flow",
        "## Demo video flow",
        "## Screenshot plan",
        "## Final submission checklist",
    ):
        assert heading in text

    for required_item in (
        "Education",
        "/feedback",
        "public YouTube",
        "under three minutes",
        "Codex",
        "GPT-5.6",
        "https://github.com/DerAndr/machine_learning_course_basics",
    ):
        assert required_item in text

    for demo in DEMOS.values():
        assert demo["live"] in text
        assert demo["offline"] in text

    assert "rewrite" in text.lower()
    assert "own voice" in text.lower()


def test_build_week_submission_setup_matches_supported_python_runtime() -> None:
    text = Path(SUBMISSION_PREPARATION_PATH).read_text(encoding="utf-8")

    assert "Python 3.12" in text
    assert "Python 3.11+" not in text
    assert "`uv` can provision" in text


def test_learning_companions_architecture_contract() -> None:
    path = Path(ARCHITECTURE_PATH)
    assert path.is_file()

    text = path.read_text(encoding="utf-8")
    for heading in (
        "# Learning Companions Architecture",
        "## What a learning companion is",
        "## Architectural layers",
        "## Responsibility boundaries",
        "## Portability model",
        "## ML-course mapping",
        "## How to use the architecture",
        "## Assurance and safety",
        "## Maintenance rules",
    ):
        assert heading in text

    for term in (
        "interactive-learning-experience-builder",
        "repository adapter",
        "experience specification",
        "grounded JSON payload",
        "deterministic",
        "self-contained",
        "file://",
        "validation",
        "student repository",
        "teacher repository",
    ):
        assert term in text

    assert "```mermaid" in text
    assert "flowchart LR" in text
    assert "| Portable core skill |" in text
    for statement in (
        "Do not create one skill per lecture or topic.",
        "The portable core remains domain-neutral.",
        "A repository adapter owns only stable local rules.",
        "Private sources are excluded at context discovery.",
    ):
        assert statement in text


def test_learning_companions_architecture_is_linked_from_repository_guides() -> None:
    for document_path, link in ARCHITECTURE_LINKS.items():
        document = Path(document_path)
        assert link in document.read_text(encoding="utf-8")

        target = link.removesuffix(")").rsplit("(", maxsplit=1)[1]
        assert (document.parent / target).is_file()

    guide_text = Path("docs/interactive-lecture-learning-assistant.md").read_text(encoding="utf-8")
    contributor_text = Path("docs/contributing-to-textbook.md").read_text(encoding="utf-8")
    assert "operational guide" in guide_text.lower()
    assert "complement" in contributor_text.lower()


def test_pages_deployment_is_limited_to_student_repository() -> None:
    workflow = Path(".github/workflows/build-textbook-preview.yml").read_text(encoding="utf-8")

    assert "github.repository == 'DerAndr/machine_learning_course_basics'" in workflow


def test_learning_companion_ci_covers_documentation_and_browser_contracts() -> None:
    required_documentation = (
        "docs/interactive-lecture-learning-assistant.md",
        "docs/contributing-to-textbook.md",
        "docs/okf-authoring-guide.md",
        "docs/learning-companions-architecture.md",
        "docs/student-quickstart.md",
        "docs/student-learning-companion-quickstart.md",
        "docs/build-week-integration-evidence.md",
        "docs/build-week-submission-preparation.md",
    )
    required_ruff_check_targets = (
        "tests/test_eda_lecture_experience.py",
        "tests/test_regression_lecture_experience.py",
        "tests/test_classification_part_1_lecture_experience.py",
    )
    required_node_tests = (
        "tests/quiz_state_machine.test.js",
        "tests/visualization_models.test.js",
        "tests/visualization_control_state.test.js",
    )

    workflow_triggers = {
        Path(".github/workflows/build-textbook-preview.yml"): ("push",),
        Path(".github/workflows/validate-okf.yml"): ("push", "pull_request"),
    }
    for workflow_path, triggers in workflow_triggers.items():
        workflow = workflow_path.read_text(encoding="utf-8")
        for trigger in triggers:
            trigger_start = workflow.index(f"  {trigger}:\n")
            paths_start = workflow.index("    paths:\n", trigger_start)
            next_trigger_match = re.search(
                r"\n  [a-z_]+:", workflow[paths_start + len("    paths:\n") :]
            )
            assert next_trigger_match is not None
            next_trigger = paths_start + len("    paths:\n") + next_trigger_match.start()
            paths_block = workflow[paths_start:next_trigger]
            for document_path in required_documentation:
                assert document_path in paths_block
        for test_path in required_ruff_check_targets:
            assert workflow.count(test_path) >= 3
        for test_path in required_node_tests:
            assert test_path in workflow


def test_build_week_evidence_records_completed_local_semantic_acceptance() -> None:
    evidence = Path("docs/build-week-integration-evidence.md").read_text(encoding="utf-8")

    for completed_check in (
        "Deterministic regeneration: completed",
        "Local full suite: completed (222 passed, 1 skipped)",
        "Validators and preview build: completed",
        "Browser acceptance: completed",
    ):
        assert completed_check in evidence

    assert "GitHub Actions | Pending for this branch." in evidence
    assert "GitHub Pages | Pending for this branch." in evidence
    canonical_browser_evidence = (
        "Browser acceptance: completed on 2026-07-20 in storage-disabled, "
        "sandboxed frames for all three companions. Scripts were allowed, but "
        "origin and `localStorage` were unavailable; each frame rendered its "
        "heading and default UI, and the browser console recorded zero warnings "
        "or errors."
    )
    assert canonical_browser_evidence in evidence
    assert (
        "Separate browser checks completed control-state retention, both palette "
        "modes, and no horizontal overflow at 390px." in evidence
    )
    assert "must be reconfirmed during final task 9" not in evidence.lower()


def test_textbook_skill_requires_mobile_quiz_contract() -> None:
    text = Path(".agents/skills/ml-course-textbook-contributor/SKILL.md").read_text()

    assert "wrong answers" in text.lower()
    assert "sticky progress" in text.lower()
    assert "mobile chrome" in text.lower()
