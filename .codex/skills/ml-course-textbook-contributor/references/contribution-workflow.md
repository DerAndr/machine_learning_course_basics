# Contribution Workflow Reference

## Source hierarchy

1. `lectures/<slug>/lecture_notes.md` for student-facing course explanations.
2. `lectures/<slug>/README.md` and `links.yaml` for scope and canonical assets.
3. `lecture_examples/` and public practical notebooks for workflow-level provenance.
4. External primary sources only when the course source is insufficient or a claim needs independent support.

## OKF page checklist

- `type`, `title`, `description`, `tags`, `timestamp`, `status` are present.
- Instructional pages have one to three `learning_objectives`.
- The title matches the first H1.
- Exactly one H1 exists.
- The description is short enough for cards and manifest use.
- Mathematical pages include equations and explain their meaning in prose.
- Practical pages include assumptions, limitations, and common failure modes.
- Relationship metadata points to existing OKF pages.
- Body links are relative Markdown links.
- `source_materials` points only to public-safe lecture paths or stable URLs.

## Textbook renderer checklist

- `tools/build_textbook_preview.py` should not become the pedagogical source of truth.
- OKF pages should remain readable on GitHub without generated HTML.
- Renderer changes need tests in `tests/test_textbook_preview.py`.
- Generated files remain under `site/_build/` and are not committed.
- Browser labs need committed source assets under `site/assets/` and public-safe data under `site/data/`.

## Agent manifest checklist

The generated `okf-manifest.json` should preserve:

- `id`
- `okf_path`
- `title`
- `description`
- `type`
- `tags`
- `learning_objectives`
- `skills`
- `prerequisites`
- `related_concepts`
- `related_labs`
- `source_materials`
- `textbook_path`

`skills` must equal `learning_objectives`.

## Review questions

Ask these before accepting a contribution:

1. Would a student learn something precise from this page without opening the full lecture?
2. Are formulas present where formulas are the clearest explanation?
3. Does every formula have an interpretation?
4. Does the page say when the idea fails or becomes misleading?
5. Can an agent discover this page through the manifest and relationships?
6. Do local and online checks prove that the rendered textbook updated?
