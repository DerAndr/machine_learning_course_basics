from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_builder():
    module_path = Path("tools/build_textbook_preview.py")
    spec = importlib.util.spec_from_file_location("build_textbook_preview", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_textbook_preview


def test_textbook_preview_builds_interactive_lab(tmp_path: Path) -> None:
    build_textbook_preview = load_builder()
    output = build_textbook_preview(output=tmp_path / "textbook")

    index = output / "index.html"
    lab = output / "labs" / "classification-threshold-explorer.html"
    contributing = output / "contributing" / "contributing-to-textbook.html"
    data = output / "data" / "classification-threshold-scores.json"
    manifest = output / "okf-manifest.json"

    assert index.is_file()
    assert lab.is_file()
    assert contributing.is_file()
    assert data.is_file()
    assert manifest.is_file()

    lab_html = lab.read_text(encoding="utf-8")
    assert "Interactive threshold explorer" in lab_html
    assert "data-threshold-lab" in lab_html
    assert "Static fallback" in lab_html
    assert "Learning route" in lab_html
    assert "Classification Metrics" in lab_html
    assert "Agent manifest" in lab_html
    assert "<h2>Skills</h2>" in lab_html
    assert "<h2>Course materials</h2>" in lab_html
    assert "Slides PDF" in lab_html
    assert "Practical assignment" in lab_html
    assert "classification_part1_practical_student_90min.ipynb" in lab_html
    assert "mathjax" in lab_html.lower()

    manifest_text = manifest.read_text(encoding="utf-8")
    assert "ml-course-okf-manifest-v1" in manifest_text
    assert "supervised-learning/classification/classification-metrics" in manifest_text
    assert "contributing/contributing-to-textbook" in manifest_text
    manifest_data = json.loads(manifest_text)
    for concept in manifest_data["concepts"]:
        assert concept["skills"] == concept["learning_objectives"]
    classification = next(
        concept
        for concept in manifest_data["concepts"]
        if concept["id"] == "supervised-learning/classification/classification"
    )
    assert (
        "/lectures/lecture_05_classification_part_1/slides/lecture.pdf"
        in classification["source_materials"]
    )
    assert (
        "/lectures/lecture_05_classification_part_1/practical_session/"
        "classification_part1_practical_student_90min.ipynb" in classification["source_materials"]
    )

    metrics_html = (
        output / "supervised-learning" / "classification" / "classification-metrics.html"
    ).read_text(encoding="utf-8")
    assert "\\mathrm{Precision}" in metrics_html
    assert "F_\\beta" in metrics_html
