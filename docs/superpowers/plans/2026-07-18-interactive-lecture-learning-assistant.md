# Interactive Lecture Learning Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repository-local skill that generates one-file offline lecture review sites, then use it to create and verify an EDA learning experience.

**Architecture:** A Python standard-library generator validates a structured JSON authoring payload and injects it into one HTML template. The generated page embeds all CSS, JavaScript, SVG interactions, quiz banks, settings, and fallbacks; a separate validator enforces the offline and accessibility contract.

**Tech Stack:** Python 3.12 standard library, pytest, YAML metadata, semantic HTML, inline CSS, vanilla JavaScript, inline SVG.

## Global Constraints

- Do not modify or delete `okf/`.
- Generated sites must open directly through `file://` with no build, server, install, account, or network dependency.
- Each quiz difficulty must contain exactly 10 questions.
- Generation defaults and in-page settings must both support difficulty, focus-friendly mode, color-blind mode, and funny topic-related break prompts.
- Charts must communicate through labels, shapes, patterns, or line styles in addition to color.
- JavaScript failure must leave explanations, quiz content, and chart fallbacks readable.
- Do not use private solutions, answer keys, grading data, or untracked quiz workbooks.

## File Map

- `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md`: agent workflow and source policy.
- `.agents/skills/ml-course-interactive-learning-assistant/agents/openai.yaml`: UI metadata.
- `.agents/skills/ml-course-interactive-learning-assistant/references/content-contract.md`: exact JSON and accessibility contract.
- `.agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html`: complete offline application shell.
- `.agents/skills/ml-course-interactive-learning-assistant/scripts/generate_lecture_site.py`: payload validation and deterministic rendering.
- `.agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py`: generated HTML checks and CLI.
- `lecture_experiences/content/lecture_01_eda.json`: reviewable EDA authoring payload.
- `lecture_experiences/lecture_01_eda/index.html`: generated portable EDA site.
- `tests/test_interactive_learning_assistant_skill.py`: skill discovery and instruction contract.
- `tests/test_lecture_site_generator.py`: generator and validator tests.
- `tests/test_eda_lecture_experience.py`: EDA content and generated-output regression tests.
- `docs/superpowers/evals/interactive-learning-assistant-baseline.md`: RED-phase baseline observation.

---

### Task 1: Establish Skill RED Test and Baseline

**Files:**
- Create: `tests/test_interactive_learning_assistant_skill.py`
- Create: `docs/superpowers/evals/interactive-learning-assistant-baseline.md`

**Interfaces:**
- Consumes: approved design spec.
- Produces: failing discovery test and recorded pre-skill gaps.

- [ ] **Step 1: Run one fresh baseline scenario without the new skill**

Prompt:

```text
Create a fully offline EDA lecture review site with explanations, interactive
graphs, three difficulty levels of ten questions, focus-friendly controls,
color-blind support, funny EDA break prompts, and accessible fallbacks. Use
only public course sources and do not modify OKF.
```

Record whether the response defines source grounding, exact quiz-bank counts, one-file offline behavior, chart fallbacks, settings persistence fallback, and validation evidence.

- [ ] **Step 2: Save exact observed gaps**

Write `docs/superpowers/evals/interactive-learning-assistant-baseline.md` with the prompt, short verbatim excerpts, and a pass/fail checklist for the six behaviors above.

- [ ] **Step 3: Write failing discovery test**

```python
from pathlib import Path

import yaml


SKILL_DIR = Path(".agents/skills/ml-course-interactive-learning-assistant")


def test_interactive_learning_assistant_skill_contract() -> None:
    skill = SKILL_DIR / "SKILL.md"
    metadata_file = SKILL_DIR / "agents/openai.yaml"

    assert skill.is_file()
    assert metadata_file.is_file()

    text = skill.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["name"] == "ml-course-interactive-learning-assistant"
    assert frontmatter["description"].startswith("Use when")
    for phrase in (
        "lecture_notes.md",
        "exactly 10",
        "file://",
        "color-blind",
        "focus-friendly",
        "Do not modify `okf/`",
        "validate_lecture_site.py",
    ):
        assert phrase in text

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert metadata["interface"]["display_name"] == "Interactive Lecture Learning Assistant"
```

- [ ] **Step 4: Confirm RED**

Run:

```powershell
uv run pytest tests/test_interactive_learning_assistant_skill.py -q
```

Expected: failure because `SKILL.md` does not exist.

- [ ] **Step 5: Commit RED evidence**

```powershell
git add tests/test_interactive_learning_assistant_skill.py docs/superpowers/evals/interactive-learning-assistant-baseline.md
git commit -m "test: define interactive learning assistant contract"
```

### Task 2: Create Minimal Discoverable Skill

**Files:**
- Create: `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md`
- Create: `.agents/skills/ml-course-interactive-learning-assistant/agents/openai.yaml`
- Create: `.agents/skills/ml-course-interactive-learning-assistant/references/content-contract.md`

**Interfaces:**
- Consumes: lecture slug and four generation defaults.
- Produces: grounded JSON payload at a user-selected path and commands for generation and validation.

- [ ] **Step 1: Create skill with exact workflow**

Frontmatter:

```yaml
---
name: ml-course-interactive-learning-assistant
description: Use when creating a self-contained interactive lecture review site with grounded explanations, interactive graphs, accessibility controls, and a 10-question knowledge quiz for this ML course.
---
```

Body must define: source hierarchy; four user choices; read-only OKF rule; content-contract reference; generator and validator commands; browser verification; refusal to use answer keys or grading data.

- [ ] **Step 2: Create UI metadata**

```yaml
interface:
  display_name: Interactive Lecture Learning Assistant
  short_description: Build an offline interactive review site for one ML lecture.
  default_prompt: Create a grounded, accessible, self-contained review site for a selected lecture.
```

- [ ] **Step 3: Define JSON contract**

Require top-level keys:

```text
meta, defaults, concepts, visualizations, quizzes, break_prompts
```

Require `foundations`, `applied`, and `challenge` arrays with exactly 10 items each. Define question fields `id`, `type`, `prompt`, `options`, `answer`, `explanation`, and `concept`. Define supported visualization types `histogram`, `boxplot`, `scatter`, and `missingness`, each with `fallback`.

- [ ] **Step 4: Run GREEN test**

```powershell
uv run pytest tests/test_interactive_learning_assistant_skill.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add .agents/skills/ml-course-interactive-learning-assistant tests/test_interactive_learning_assistant_skill.py
git commit -m "feat: add interactive lecture learning assistant skill"
```

### Task 3: Build Generator and Validator with TDD

**Files:**
- Create: `tests/test_lecture_site_generator.py`
- Create: `.agents/skills/ml-course-interactive-learning-assistant/scripts/generate_lecture_site.py`
- Create: `.agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py`
- Create: `.agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html`

**Interfaces:**
- Consumes: `generate_site(content_path: Path, template_path: Path, output_path: Path) -> Path`.
- Produces: `validate_payload(payload: dict[str, object]) -> list[str]`, `render_site(template: str, payload: dict[str, object]) -> str`, and `validate_html(path: Path) -> list[str]`.

- [ ] **Step 1: Write failing generator tests**

Test that valid payload rendering:

```python
html = render_site("<script>const CONTENT = __CONTENT_JSON__;</script>", payload)
assert "__CONTENT_JSON__" not in html
assert json.dumps(payload, ensure_ascii=False, separators=(",", ":")) in html
```

Test failures for 9-question banks, missing explanations, missing graph fallback, unsupported visualization types, and templates with zero or two `__CONTENT_JSON__` markers.

- [ ] **Step 2: Write failing HTML validator tests**

Validate rejection of `https://` script/style/image/font resources, missing settings controls, missing `<noscript>`, missing `prefers-reduced-motion`, missing visible focus rule, and absent graph fallback.

- [ ] **Step 3: Confirm RED**

```powershell
uv run pytest tests/test_lecture_site_generator.py -q
```

Expected: collection failure because generator modules do not exist.

- [ ] **Step 4: Implement payload validation and rendering**

Use deterministic compact JSON:

```python
encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
return template.replace("__CONTENT_JSON__", encoded)
```

Return all payload problems together. Raise `ValueError("\n".join(errors))` before writing output.

- [ ] **Step 5: Implement HTML validation CLI**

Use `html.parser.HTMLParser` plus explicit text checks. CLI:

```powershell
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py <index.html>
```

Exit `0` with `VALID: <path>` or exit `1` with one `ERROR:` line per violation.

- [ ] **Step 6: Create minimal template contract**

Template must contain `__CONTENT_JSON__` exactly once plus:

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
<main id="main-content"></main>
<noscript><section id="static-content"></section></noscript>
<script>const CONTENT = __CONTENT_JSON__;</script>
```

It must also include inline focus-visible and reduced-motion styles.

- [ ] **Step 7: Run GREEN tests**

```powershell
uv run pytest tests/test_lecture_site_generator.py -q
uv run ruff check .agents/skills/ml-course-interactive-learning-assistant/scripts tests/test_lecture_site_generator.py
```

Expected: all pass.

- [ ] **Step 8: Commit**

```powershell
git add .agents/skills/ml-course-interactive-learning-assistant tests/test_lecture_site_generator.py
git commit -m "feat: add offline lecture site generator"
```

### Task 4: Implement Interactive Template

**Files:**
- Modify: `.agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html`
- Modify: `tests/test_lecture_site_generator.py`

**Interfaces:**
- Consumes: validated content payload embedded as `CONTENT`.
- Produces: in-page settings, four SVG visualization renderers, static fallbacks, quiz engine, score review, retry, and local-storage fallback.

- [ ] **Step 1: Add failing template behavior checks**

Assert presence of stable hooks:

```text
data-setting="difficulty"
data-setting="focus"
data-setting="color-blind"
data-setting="break-prompts"
renderHistogram
renderBoxplot
renderScatter
renderMissingness
renderQuiz
showQuizResults
safeStorage
aria-live="polite"
```

- [ ] **Step 2: Confirm RED**

Run:

```powershell
uv run pytest tests/test_lecture_site_generator.py -q
```

Expected: hook assertions fail.

- [ ] **Step 3: Implement minimal responsive interface**

Create header, setup controls, progress bar, concept cards, visualization controls, quiz panel, source notes, and non-blocking break-prompt region. Use CSS custom properties for normal and color-blind-safe palettes. Use text labels and SVG pattern/shape differences in addition to color.

- [ ] **Step 4: Implement state and persistence**

State fields:

```javascript
const state = {
  difficulty: CONTENT.defaults.difficulty,
  focus: CONTENT.defaults.focus_mode,
  colorBlind: CONTENT.defaults.color_blind,
  breakPrompts: CONTENT.defaults.break_prompts,
  conceptIndex: 0,
  quizIndex: 0,
  responses: [],
};
```

`safeStorage` catches all `localStorage` reads and writes and keeps in-memory state on failure.

- [ ] **Step 5: Implement four SVG renderers**

Every renderer updates a nearby summary and retains payload-provided fallback table/text in HTML. Slider/input changes update SVG, labels, and summary without moving keyboard focus.

- [ ] **Step 6: Implement quiz engine**

Render one question at a time in focus mode. Provide immediate feedback only after submission, then enable Next. Results list every question, learner response, correct answer, and explanation. Retry clears responses but preserves settings.

- [ ] **Step 7: Run tests and validator**

```powershell
uv run pytest tests/test_lecture_site_generator.py -q
uv run ruff check .agents/skills/ml-course-interactive-learning-assistant/scripts tests/test_lecture_site_generator.py
```

Expected: all pass.

- [ ] **Step 8: Commit**

```powershell
git add .agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html tests/test_lecture_site_generator.py
git commit -m "feat: add accessible lecture site interactions"
```

### Task 5: Author and Generate EDA Reference Experience

**Files:**
- Create: `tests/test_eda_lecture_experience.py`
- Create: `lecture_experiences/content/lecture_01_eda.json`
- Create: `lecture_experiences/lecture_01_eda/index.html`

**Interfaces:**
- Consumes: public Lecture 01 notes, README, examples README, practical README, and optional read-only OKF material.
- Produces: one reviewable JSON payload and one portable generated page.

- [ ] **Step 1: Write failing EDA content test**

Assert:

```python
assert payload["meta"]["lecture_slug"] == "lecture_01_eda"
assert {item["type"] for item in payload["visualizations"]} == {
    "histogram", "boxplot", "scatter", "missingness"
}
assert all(len(payload["quizzes"][level]) == 10 for level in levels)
assert len({q["id"] for level in levels for q in payload["quizzes"][level]}) == 30
assert all("lectures/lecture_01_eda/" in source for source in payload["meta"]["sources"])
assert all("okf/" not in changed_path for changed_path in git_diff_paths)
```

- [ ] **Step 2: Confirm RED**

```powershell
uv run pytest tests/test_eda_lecture_experience.py -q
```

Expected: failure because payload and generated page do not exist.

- [ ] **Step 3: Author grounded EDA payload**

Cover:

- EDA before modeling;
- data types and structure;
- center, spread, and skew;
- histogram bin-width effects;
- IQR and box-plot outliers;
- scatter association versus causation;
- missing counts versus proportions;
- automation as a starting point, not a substitute for reasoning.

Write 30 unique questions: Foundations tests definitions and recognition; Applied uses short data scenarios; Challenge tests interpretation, failure modes, and misleading displays. Every answer gets a concise explanation and concept tag.

- [ ] **Step 4: Generate site**

```powershell
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_lecture_site.py `
  --content lecture_experiences/content/lecture_01_eda.json `
  --template .agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html `
  --output lecture_experiences/lecture_01_eda/index.html
```

Expected: `GENERATED: lecture_experiences/lecture_01_eda/index.html`.

- [ ] **Step 5: Validate site**

```powershell
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py lecture_experiences/lecture_01_eda/index.html
uv run pytest tests/test_eda_lecture_experience.py -q
```

Expected: validator prints `VALID`; tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lecture_experiences tests/test_eda_lecture_experience.py
git commit -m "feat: add interactive EDA learning experience"
```

### Task 6: Scenario Re-Test and Full Verification

**Files:**
- Modify: `docs/superpowers/evals/interactive-learning-assistant-baseline.md`
- Modify: `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md` only if post-skill testing exposes a concrete gap.

**Interfaces:**
- Consumes: completed skill and EDA output.
- Produces: behavioral evidence and final verification results.

- [ ] **Step 1: Re-run original scenario with the skill**

Use a fresh context containing the skill but not the intended answer. Record whether all six baseline gaps are corrected.

- [ ] **Step 2: Fix only observed skill gaps**

Add precise positive requirements to `SKILL.md`; repeat the same scenario until it consistently produces the required workflow.

- [ ] **Step 3: Run static and repository checks**

```powershell
uv run ruff format --check .agents/skills/ml-course-interactive-learning-assistant/scripts tests/test_interactive_learning_assistant_skill.py tests/test_lecture_site_generator.py tests/test_eda_lecture_experience.py
uv run ruff check .agents/skills/ml-course-interactive-learning-assistant/scripts tests/test_interactive_learning_assistant_skill.py tests/test_lecture_site_generator.py tests/test_eda_lecture_experience.py
uv run pytest
```

Expected: all checks pass.

- [ ] **Step 4: Run browser smoke test**

Open the generated `index.html` directly. Verify all four settings, each graph control, one quiz completion, results review, retry, keyboard navigation, and storage-disabled fallback. Confirm browser console has no errors.

- [ ] **Step 5: Prove OKF remained unchanged**

```powershell
git diff 00f7a60 --name-only -- okf
```

Expected: no output.

- [ ] **Step 6: Commit final evidence**

```powershell
git add .agents/skills/ml-course-interactive-learning-assistant docs/superpowers/evals/interactive-learning-assistant-baseline.md
git commit -m "test: verify interactive learning assistant workflow"
```
