# OKF Authoring Guide

## Purpose

The `okf/` directory is a concise, independently conformant Open Knowledge Format v0.1 bundle. It complements the full course in `lectures/`; it does not replace or duplicate it. This student repository is the source of truth for public-safe OKF content.

The long-term product shape is an interactive textbook. Concepts should be useful as plain Markdown, but they should also carry enough structure for a renderer to build guided navigation, browser-based labs, search, and accessible fallbacks.

## Before writing

1. Identify one learner question and one practical outcome.
2. Map the source lecture sections, slides, examples, and notebooks.
3. Check whether another concept already owns the idea.
4. Confirm that every source and link is safe for the intended public bundle.

## Concept template

```markdown
---
type: ML Algorithm
title: K-Nearest Neighbors
description: A distance-based classifier that predicts labels from nearby training observations.
tags: [classification, supervised-learning, algorithm]
timestamp: 2026-06-22T00:00:00Z
status: draft
learning_objectives:
  - Explain how KNN predicts a class.
difficulty: introductory
estimated_reading_minutes: 5
prerequisites:
  - /foundations/features-and-targets.md
related_labs:
  - /labs/knn-decision-boundary.md
source_materials:
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
---

# Core idea

Write a concise explanation in original language.

# Go deeper

Link to the full lecture or notebook with a standard relative Markdown link when the target is inside the bundle.
```

## Required metadata

Every concept requires `type`, `title`, `description`, `tags`, `timestamp`, and `status`. Instructional concepts also require one to three `learning_objectives`. Use these statuses: `draft`, `review`, `published`, or `deprecated`.

Unknown metadata fields and types remain valid under OKF. Add a controlled type only when it materially improves filtering or presentation.

## Links

- Use relative Markdown links in concept bodies and indexes.
- Use bundle-root paths in `prerequisites`, `related_concepts`, and `related_labs`.
- Use repository-root `/lectures/...` paths or stable external URLs in `source_materials`.
- Treat published concept paths as stable identifiers. Record a migration decision before moving one.
- Never reference non-public solutions, answer keys, private drafts, raw migration sources, or other non-public materials from the production bundle.

## Indexes

Every populated directory should have an `index.md`. Nested indexes have no frontmatter. The root index may contain only `okf_version: "0.1"` frontmatter.

Use this entry format:

```markdown
* [Display title](relative-path/) - One-sentence description.
```

Keep pedagogical sequences in teaching order. Use alphabetical order only for reference collections.

## Content and citations

- Target 3-8 minutes of reading and roughly 400-1,200 words.
- Prefer one central idea, one example, and one or two equations at most.
- Link to notebooks for full workflows.
- Put external claim support in a numbered `# Citations` section.
- Use `source_materials` for course provenance; it is not a substitute for citations.
- Do not copy long passages, slide narration, or classroom administration text.

## Interactive textbook contract

- Keep durable pedagogy in `okf/`: concepts, objectives, prerequisites, lab descriptions, and links.
- Keep renderer code, generated pages, JavaScript components, and static lab datasets outside `okf/`.
- Describe each interactive lab as a first-class OKF concept with inputs, outputs, learning objective, source data policy, and fallback behavior.
- Every interactive lab must have a no-JavaScript fallback such as a static image, precomputed table, or short explanation.
- The first version must run entirely in the browser: no accounts, persistence, server-side Python, arbitrary code execution, or uploaded student data.
- Interactive widgets should clarify one learner decision or mechanism; avoid dashboards that expose many controls without a teaching sequence.

## Validate

```bash
uv run python tools/validate_okf.py okf/
uv run pytest tests/test_okf_validation.py
```

Errors are mechanical and block integration. Warnings identify editorial review items such as orphan concepts or heading structure.
