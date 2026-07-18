---
name: ml-course-textbook-contributor
description: Improve the ML course interactive textbook in the student repository. Use when adding or revising OKF concepts, learning paths, browser labs, textbook rendering, agent manifests, contribution guides, or public-safe course knowledge modules. Enforce source-grounded pedagogy, formula-level rigor when appropriate, OKF validation, rendered textbook checks, and consistency between descriptions, skills, learning objectives, indexes, and the agent manifest.
---

# ML Course Textbook Contributor

Use this skill to improve the interactive textbook as both a student-facing learning resource and an agent-readable knowledge system.

## Core rule

Treat `okf/` as the source of truth. The rendered textbook, `okf-manifest.json`, navigation cards, and agent-facing skills must be generated from or synchronized with OKF metadata and content.

## Contribution workflow

1. Read the relevant source lecture material under `lectures/`.
2. Read the target OKF pages and nearby indexes.
3. Decide whether the change is a concept, metric, algorithm, lab, learning path, renderer improvement, or contribution workflow improvement.
4. Write original educational text. Do not summarize superficially.
5. Include formulas, assumptions, edge cases, and failure modes when the topic needs them.
6. Keep `description` and `learning_objectives` in frontmatter precise; these drive indexes, rendered skills, and the manifest.
7. Update directory `index.md` entries with descriptions that exactly match target frontmatter.
8. Update relationships: `prerequisites`, `related_concepts`, and `related_labs`.
9. If navigation, contribution policy, or workflow changes, update README files and guides together.
10. Run validation and preview checks.
11. Push changes and verify the deployed textbook online.

## Quality bar

Reject shallow content. A good contribution should usually include:

- a clear learner question;
- a formal definition or algorithmic rule when relevant;
- formulas for mathematical topics;
- practical interpretation of every formula;
- examples or counterexamples;
- assumptions and limitations;
- links to related concepts or labs;
- public-safe provenance in `source_materials`;
- links from the rendered textbook page to lecture notes, slides PDF, and practical assignments when those course assets exist.

For interactive labs, require:

- one focused learning mechanism;
- fixed public-safe data or deterministic generated data;
- no accounts, uploads, persistence, server-side Python, or arbitrary code execution;
- a no-JavaScript fallback;
- tests that confirm the rendered lab exists and exposes the intended interaction.

For interactive learning experiences and browser labs, also require:

- wrong answers to keep the learner on the current question, preserve an
  explanation, and never mark the question complete or reveal the next one;
- changing an answer to clear stale feedback before another submission;
- whole-quiz Retry to preserve learner settings while resetting question state,
  feedback, results, and progress;
- sticky progress that remains visible but never obscures focused controls;
- mobile Chrome verification with a touch viewport, including retry, focus, and
  scrolling behavior around the sticky progress panel.

## Consistency rules

- Do not author `skills` separately. Agent-facing `skills` are generated from `learning_objectives`.
- Do not let index descriptions drift. They must match target frontmatter `description`.
- Do not put contribution planning boards into Git unless they are student-facing.
- Do not reference non-public solutions, answer keys, private drafts, raw migration sources, or other non-public course materials.
- Keep generated output under `site/_build/` ignored.
- Keep `README.md`, `AGENTS.md`, `docs/contributing-to-textbook.md`, and `docs/okf-authoring-guide.md` aligned when contribution workflow changes.

## Required commands

Run these before committing:

```powershell
uv run ruff format --check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py tests/test_okf_validation.py tests/test_textbook_preview.py tests/test_smoke.py
uv run ruff check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py tests/test_okf_validation.py tests/test_textbook_preview.py tests/test_smoke.py
uv run ty check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
```

After pushing to `main`, verify:

- GitHub Actions succeeded.
- The affected textbook pages return HTTP 200.
- The rendered page contains the expected formulas or interaction hooks.
- `okf-manifest.json` contains the concept and `skills == learning_objectives`.

## Source hierarchy

1. `lectures/<slug>/lecture_notes.md` for student-facing course explanations.
2. `lectures/<slug>/README.md` and `links.yaml` for scope and canonical assets.
3. `lecture_examples/` and public practical notebooks for workflow-level provenance.
4. `docs/` only when the page is about repository workflow, textbook contribution, or OKF authoring.
5. External primary sources only when the course source is insufficient or a claim needs independent support.

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
- `source_materials` points only to public-safe lecture paths, public documentation paths, or stable URLs.
- Lecture-backed pages link to lecture overview, lecture notes, slides PDF, practical assignment README, and student practical notebook when those files exist.

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
7. If navigation or contribution workflow changed, were README files, guides, and the agent skill updated together?
