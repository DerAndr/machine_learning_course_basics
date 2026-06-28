from __future__ import annotations

from pathlib import Path

from mlcourse.okf_validation import parse_markdown, validate_bundle


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def root_index(bundle: Path, body: str = "# Test Bundle") -> None:
    write(bundle / "index.md", f'---\nokf_version: "0.1"\n---\n\n{body}')


def concept(
    bundle: Path,
    relative: str = "concept.md",
    *,
    extra: str = "",
    body: str = "# Core idea\n\nA concise explanation.",
    concept_type: str = "Concept",
) -> Path:
    path = bundle / relative
    write(
        path,
        f"""
---
type: {concept_type}
title: Test Concept
description: A concise test concept.
tags: [foundations]
timestamp: 2026-06-22T00:00:00Z
status: draft
learning_objectives:
  - Explain the test concept.
{extra}---

{body}
""",
    )
    return path


def codes(result) -> set[str]:
    return {item.code for item in result.diagnostics}


def test_repository_scaffold_validates_without_findings() -> None:
    result = validate_bundle(Path("okf"), repository_root=Path.cwd())
    assert result.errors == ()
    assert result.warnings == ()
    assert result.index_count == 5
    assert result.concept_count == 5


def test_parse_markdown_preserves_unknown_fields() -> None:
    parsed = parse_markdown("---\ntype: Concept\ncustom: value\n---\n# Body\n")
    assert parsed.metadata == {"type": "Concept", "custom": "value"}
    assert parsed.body == "# Body\n"


def test_missing_frontmatter_and_root_index_are_errors(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    write(bundle / "concept.md", "# Missing metadata")
    result = validate_bundle(bundle)
    assert {"OKF001", "OKF009"} <= codes(result)


def test_required_fields_and_status_are_checked(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [Broken](broken.md) - Broken concept.")
    write(bundle / "broken.md", "---\ntype: Concept\nstatus: unknown\n---\n# Broken")
    result = validate_bundle(bundle)
    assert "OKF002" in codes(result)
    assert "OKF005" in codes(result)
    assert "OKF008" in codes(result)


def test_malformed_metadata_types_are_reported_without_crashing(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [Broken](broken.md) - Broken concept.")
    write(
        bundle / "broken.md",
        """
---
type: [Concept]
title: [Bad]
description: {}
tags: [Bad Tag]
timestamp: 2026-06-22
status: [draft]
learning_objectives:
  - Explain the broken concept.
---

# Broken
""",
    )
    result = validate_bundle(bundle)
    assert {"OKF004", "OKF005", "OKF006", "OKF007", "OKF024"} <= codes(result)


def test_unknown_type_is_a_warning_not_an_error(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [Custom](custom.md) - Custom concept.")
    concept(bundle, "custom.md", concept_type="Domain Note")
    result = validate_bundle(bundle)
    assert not result.errors
    assert "OKF040" in codes(result)


def test_nested_index_rejects_frontmatter_and_bad_entry(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [Area](area/) - Test area.")
    write(
        bundle / "area" / "index.md",
        "---\ntitle: Not allowed\n---\n\n# Area\n\n* [Item](item.md)",
    )
    result = validate_bundle(bundle)
    assert {"OKF012", "OKF013"} <= codes(result)


def test_relationship_paths_resolve_from_bundle_root(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [First](first.md) - First concept.")
    concept(bundle, "first.md", extra="related_concepts:\n  - /second.md\n")
    concept(bundle, "second.md", concept_type="Reference", extra="learning_objectives: null\n")
    result = validate_bundle(bundle)
    assert "OKF021" not in codes(result)
    assert "OKF042" not in codes(result)


def test_body_links_must_resolve_and_prefer_relative_paths(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [First](first.md) - First concept.")
    concept(bundle, "first.md", body="# Core idea\n\nSee [missing](/missing.md).")
    result = validate_bundle(bundle)
    assert {"OKF020", "OKF021"} <= codes(result)


def test_local_links_must_not_escape_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [First](area/first.md) - First concept.")
    concept(bundle, "area/first.md", body="# Core idea\n\nSee [outside](../../teacher.md).")
    write(tmp_path / "teacher.md", "# Private")
    result = validate_bundle(bundle)
    assert "OKF023" in codes(result)


def test_links_inside_fenced_code_are_ignored(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [First](first.md) - First concept.")
    concept(
        bundle,
        "first.md",
        body="# Core idea\n\n```markdown\n[example](missing.md)\n```\n",
    )
    result = validate_bundle(bundle)
    assert "OKF021" not in codes(result)


def test_teacher_only_sources_are_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle, "# Test\n\n* [Unsafe](unsafe.md) - Unsafe concept.")
    concept(
        bundle,
        "unsafe.md",
        extra=(
            "source_materials:\n"
            "  - /lectures/lecture_05/practical_session/example_teacher_90min.ipynb\n"
        ),
    )
    result = validate_bundle(bundle)
    assert "OKF018" in codes(result)


def test_orphan_and_citation_findings_are_warnings(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle)
    concept(bundle, "orphan.md", extra="citation_required: true\n")
    result = validate_bundle(bundle)
    assert not result.errors
    assert {"OKF041", "OKF042"} <= codes(result)


def test_orphan_cycles_are_not_reachable_without_root_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle)
    concept(bundle, "a.md", extra="related_concepts:\n  - /b.md\n")
    concept(bundle, "b.md", extra="related_concepts:\n  - /a.md\n")
    result = validate_bundle(bundle)
    orphan_paths = {item.path for item in result.diagnostics if item.code == "OKF042"}
    assert orphan_paths == {"a.md", "b.md"}


def test_log_headings_must_be_valid_descending_dates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle)
    write(bundle / "log.md", "# Log\n\n## Yesterday\n\nBad heading.\n\n## 2026-02-30\n\nBad date.")
    result = validate_bundle(bundle)
    assert "OKF051" in codes(result)


def test_diagnostics_have_deterministic_order(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    root_index(bundle)
    write(bundle / "z.md", "# Z")
    write(bundle / "a.md", "# A")
    result = validate_bundle(bundle)
    assert list(result.diagnostics) == sorted(result.diagnostics)
