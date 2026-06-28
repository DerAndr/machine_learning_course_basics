# Contributing to the Interactive Textbook

The interactive textbook is built from the Open Knowledge Format bundle in `okf/`. It is meant for both humans and agents: students should be able to read and improve concepts, while agents should be able to navigate the same knowledge through metadata, links, and `okf-manifest.json`.

Read the current preview online:

<https://derandr.github.io/machine_learning_course_basics/>

## Fast path

For a small content improvement:

1. Find the target concept in `okf/`.
2. Read the relevant lecture source in `lectures/`.
3. Improve the page, keeping the first H1 equal to the frontmatter `title`.
4. If the frontmatter `description` changes, update every index card that links to the page with the exact same sentence.
5. Run validation and rebuild the preview.
6. Check the rendered page locally or after GitHub Pages deploys.

## What to contribute

Good contributions include:

- improving an existing concept with clearer definitions, formulas, examples, and limitations;
- adding a missing concept from a lecture;
- adding or improving a learning path;
- adding a small browser-based lab with a no-JavaScript fallback;
- improving textbook rendering without moving pedagogy out of `okf/`;
- improving validation when a drift pattern is discovered.
- improving README files and guides when navigation or contribution workflow changes.

Avoid drive-by wording changes that do not improve learning.

## Source of truth

Use this hierarchy:

1. `okf/` is the source for concise textbook concepts and relationships.
2. `lectures/` is the source for full course material and provenance.
3. `site/assets/` and `site/data/` support browser labs.
4. `site/_build/` is generated output and must not be committed.

Descriptions and skills must not fork into separate versions. A page's card description comes from frontmatter `description`; agent-facing `skills` are generated from `learning_objectives`.

In short: skills are generated from `learning_objectives`.

Public documentation under `docs/` can be used as OKF provenance when the concept is about the textbook or repository workflow. Course concepts should normally cite lecture material under `lectures/`.

## Quality bar

A concept should answer one learner question well. For technical ML topics, that usually means:

- define the object precisely;
- include formulas when they are the natural language of the topic;
- explain each formula in prose;
- include assumptions and failure modes;
- connect to related concepts, labs, and lecture sources;
- avoid copying lecture prose wholesale.

For example, a page about classification metrics should not merely say "precision and recall are useful." It should define $TP$, $FP$, $TN$, $FN$, give the formulas for precision and recall, and explain when each metric matters.

## OKF frontmatter

Every concept page needs:

```yaml
type: Concept
title: Example Concept
description: One concise sentence used by indexes, cards, and the manifest.
tags: [classification, supervised-learning]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - State the skill a learner should gain.
difficulty: introductory
estimated_reading_minutes: 5
source_materials:
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
```

Instructional `learning_objectives` become agent-facing `skills`, so write them as concrete learner capabilities.

Each concept page must have exactly one H1, and that H1 must match frontmatter `title`.

## Indexes and links

- Every populated OKF directory has an `index.md`.
- Index entries use:

```markdown
* [Title](page.md) - Exact frontmatter description.
```

- Body links use relative Markdown paths.
- Relationship metadata uses bundle-root paths such as `/supervised-learning/classification/classification.md`.

The validator fails if an index description differs from the target page description.

## Interactive labs

A lab should teach one mechanism. It must include:

- fixed public-safe data;
- browser-only interaction;
- no accounts, uploads, persistence, server-side Python, or arbitrary code execution;
- a static fallback for readers without JavaScript;
- tests that confirm the lab renders.

## Agent skill

Agents improving the textbook should use the repo skill:

```text
.codex/skills/ml-course-textbook-contributor/SKILL.md
```

That skill summarizes the contribution workflow and points to a reference checklist.

When the workflow itself changes, update the skill, this guide, `docs/okf-authoring-guide.md`, and any affected README navigation together.

## Local checks

Run before committing:

```powershell
uv run ruff format --check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py tests/test_okf_validation.py tests/test_textbook_preview.py tests/test_smoke.py
uv run ruff check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py tests/test_okf_validation.py tests/test_textbook_preview.py tests/test_smoke.py
uv run ty check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
```

## Online checks

After merge to `main`, check:

- GitHub Actions succeeded.
- The changed textbook page opens on GitHub Pages.
- The page contains the expected formulas, examples, or interaction.
- `https://derandr.github.io/machine_learning_course_basics/okf-manifest.json` includes the updated concept.
