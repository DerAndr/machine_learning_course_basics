---
name: ml-course-interactive-learning-assistant
description: Use when creating a self-contained interactive lecture review site with grounded explanations, interactive graphs, accessibility controls, and a 10-question knowledge quiz for this ML course.
---

# Interactive Lecture Learning Assistant

Create one deterministic, portable `index.html` that opens through `file://`
without a server, network request, external font, CDN, account, or runtime
dependency. Embed all content, styles, scripts, SVG, fallbacks, and quiz banks.
Keep explanations and full quiz content statically readable if JavaScript fails.

## Workflow

1. Resolve the lecture slug through `lectures/index.yaml`.
2. Ask the user to choose four generation defaults:
   - quiz depth: Foundations, Applied, or Challenge;
   - focus-friendly mode: on or off;
   - color-blind-safe palette: on or off;
   - funny topic-related break prompts: on or off.
3. Ground content in this source order:
   1. `lectures/<slug>/lecture_notes.md`;
   2. `lectures/<slug>/README.md` and `links.yaml`;
   3. public `lecture_examples/` material;
   4. the public practical README and student notebook;
   5. relevant OKF concepts as read-only supporting sources.
4. Do not modify `okf/`. Do not use private solutions, answer keys, grading data,
   or untracked quiz workbooks.
5. Read [references/content-contract.md](references/content-contract.md), then
   write a grounded JSON payload to the user-selected path. Name every course
   source used. Fix unsupported claims, ambiguous answers, and missing evidence
   before generation.
6. Generate the site:

   ```powershell
   uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_lecture_site.py `
     --content <content.json> `
     --template .agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html `
     --output <output-directory>/index.html
   ```

7. Validate the generated file:

   ```powershell
   uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py <output-directory>/index.html
   ```

8. Open the generated `index.html` directly through `file://`. Verify all four
   settings, every graph control and fallback, one exactly 10-question quiz,
   answer review, retry, keyboard navigation, visible focus, reduced motion,
   storage-disabled behavior, and a clean browser console.
