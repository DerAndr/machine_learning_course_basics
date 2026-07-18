---
name: ml-course-interactive-learning-assistant
description: Use when creating an interactive review for an ML-course lecture and its course-specific source, safety, and publishing rules are needed.
---

# ML Course Interactive Learning Assistant

This is the ML-course adapter for
`interactive-learning-experience-builder`. Use that portable core for the
experience specification, content contract, single-file rendering, accessibility
requirements, and reusable validation. This adapter supplies only stable
ML-course constraints.

## Course context

- Read repository instructions and relevant contributor guidance before selecting
  sources.
- Resolve the requested lecture slug through `lectures/index.yaml`.
- Store the grounded payload at `lecture_experiences/content/<lecture_slug>.json`
  and the generated review at
  `lecture_experiences/<lecture_slug>/index.html`.
- The committed offline HTML is the source for the Pages copy. Build tooling
  copies it to `site/_build/demos/<lecture_slug>/index.html`; do not maintain a
  second demo under `site/`.

## Source and safety policy

Ground claims in this order:

1. `lectures/<lecture_slug>/lecture_notes.md`;
2. `lectures/<lecture_slug>/README.md` and `links.yaml`;
3. public `lecture_examples/` material;
4. the public practical README and student notebook;
5. relevant `okf/` concepts as read-only supporting sources.

Do not modify `okf/`. Do not use private solutions, teacher notebooks, answer
keys, grading data, unpublished drafts, or untracked quiz workbooks. Name every
public course source used in the payload.

## Recurring generation workflow

1. Ask the learner for four defaults: Foundations, Applied, or Challenge quiz
   depth; focus-friendly mode on or off; color-blind-safe palette on or off; and
   funny topic-related break prompts on or off.
2. Create the short experience specification required by
   `interactive-learning-experience-builder`, then ground the payload using the
   course source order above.
3. Include the three `foundations`, `applied`, and `challenge` quiz banks with
   exactly 10 questions each. Embed break prompts even when their initial display
   setting is off.
4. Generate with the portable core:

   ```powershell
   uv run python .agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py `
     --content lecture_experiences/content/<lecture_slug>.json `
     --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
     --output lecture_experiences/<lecture_slug>/index.html
   ```

5. Validate with the portable core:

   ```powershell
   uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
     lecture_experiences/<lecture_slug>/index.html
   ```

6. Open the generated `index.html` directly through `file://`. Verify all four
   learner settings, visualizations and their fallbacks, answer review and
   whole-quiz Retry, keyboard navigation, visible focus, reduced motion, and
   storage fallback when storage is disabled. The core remains responsible for
   deterministic single-file output, static no-JavaScript explanations and quiz
   review, and accessible chart fallbacks.

## Course checks and deployment

Before committing a lecture review, run the focused experience tests and the
course preview build:

```powershell
uv run pytest tests/test_eda_lecture_experience.py tests/test_interactive_learning_assistant_skill.py tests/test_interactive_learning_assistant_docs.py -q
uv run python tools/build_textbook_preview.py
```

After merging to `main`, verify GitHub Actions and the deployed Pages review in
the public student repository.
