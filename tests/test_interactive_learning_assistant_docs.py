from pathlib import Path

LIVE_URL = "https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/"
OFFLINE_PATH = "lecture_experiences/lecture_01_eda/index.html"
SKILL_PATH = ".agents/skills/ml-course-interactive-learning-assistant/SKILL.md"


def test_learning_assistant_documentation_is_discoverable() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/interactive-lecture-learning-assistant.md")
    lecture = Path("lectures/lecture_01_eda/README.md").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert guide.is_file()
    guide_text = guide.read_text(encoding="utf-8")
    assert LIVE_URL in readme
    assert LIVE_URL in guide_text
    assert OFFLINE_PATH in readme
    assert OFFLINE_PATH in guide_text
    assert OFFLINE_PATH in lecture
    assert SKILL_PATH in guide_text
    assert "lecture_experiences/content/" in agents
    assert "validate_lecture_site.py" in agents


def test_pages_deployment_is_limited_to_student_repository() -> None:
    workflow = Path(".github/workflows/build-textbook-preview.yml").read_text(encoding="utf-8")

    assert "github.repository == 'DerAndr/machine_learning_course_basics'" in workflow
