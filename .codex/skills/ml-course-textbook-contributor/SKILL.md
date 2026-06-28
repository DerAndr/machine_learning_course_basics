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
9. Run validation and preview checks.
10. Push changes and verify the deployed textbook online.

## Quality bar

Reject shallow content. A good contribution should usually include:

- a clear learner question;
- a formal definition or algorithmic rule when relevant;
- formulas for mathematical topics;
- practical interpretation of every formula;
- examples or counterexamples;
- assumptions and limitations;
- links to related concepts or labs;
- public-safe provenance in `source_materials`.

For interactive labs, require:

- one focused learning mechanism;
- fixed public-safe data or deterministic generated data;
- no accounts, uploads, persistence, server-side Python, or arbitrary code execution;
- a no-JavaScript fallback;
- tests that confirm the rendered lab exists and exposes the intended interaction.

## Consistency rules

- Do not author `skills` separately. Agent-facing `skills` are generated from `learning_objectives`.
- Do not let index descriptions drift. They must match target frontmatter `description`.
- Do not put contribution planning boards into Git unless they are student-facing.
- Do not reference non-public solutions, answer keys, private drafts, raw migration sources, or other non-public course materials.
- Keep generated output under `site/_build/` ignored.

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

## References

Read `references/contribution-workflow.md` when planning a non-trivial textbook contribution or reviewing another agent's work.
