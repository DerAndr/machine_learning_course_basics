# Portable Learning Experience and Mobile Quiz Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the interactive learning builder portable across repositories and knowledge bases, preserve the ML-course adapter and OKF rules, and fix mobile quiz retries, stale feedback, and sticky progress.

**Architecture:** Move reusable assets and scripts into a domain-neutral core skill. Keep the ML-course skill as a thin repository adapter and keep textbook contribution rules separate. Use an explicit quiz state machine that records first-attempt accuracy while allowing retries until correct.

**Tech Stack:** Codex skills, Python 3.12, pytest, vanilla HTML/CSS/JavaScript, Chrome/Playwright device emulation.

## Global Constraints

- The core must not assume ML, lectures, OKF, `uv`, Python, GitHub Pages, or a particular directory layout.
- Generated experiences remain deterministic, fully self-contained, offline-capable single HTML files.
- Repository instructions, build tooling, and publishing commands are optional.
- Wrong answers never complete or advance a question.
- Results report first-attempt correctness and total attempts.
- Whole-quiz Retry preserves settings and resets every quiz-state field.
- The repository copy is canonical; the global installation is a deterministic sync.
- Existing ML-course and OKF behavior must remain intact.

---

### Task 1: Create the portable core skill

**Files:**
- Create: `.agents/skills/interactive-learning-experience-builder/SKILL.md`
- Create: `.agents/skills/interactive-learning-experience-builder/agents/openai.yaml`
- Create: `.agents/skills/interactive-learning-experience-builder/references/context-discovery.md`
- Create: `.agents/skills/interactive-learning-experience-builder/references/content-contract.md`
- Create: `.agents/skills/interactive-learning-experience-builder/references/repository-adapter-template.md`
- Move: `.agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html` to `.agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html`
- Move: `.agents/skills/ml-course-interactive-learning-assistant/scripts/generate_lecture_site.py` to `.agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py`
- Move: `.agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py` to `.agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py`
- Create: `tests/test_interactive_learning_experience_builder_skill.py`
- Modify: `tests/test_lecture_site_generator.py`

**Interfaces:**
- Consumes: a JSON payload with `meta.experience_id`, named sources, concepts, visualizations, three quiz banks, and break prompts.
- Produces: `validate_payload(payload, repository_root=None) -> list[str]`, `generate_site(payload, template) -> str`, and `validate_html(path) -> list[str]`.

- [ ] **Step 1: Write failing portability and skill-contract tests**

```python
CORE = Path(".agents/skills/interactive-learning-experience-builder")

def test_core_skill_is_domain_neutral() -> None:
    text = (CORE / "SKILL.md").read_text(encoding="utf-8")
    for forbidden in ("lectures/index.yaml", "okf/", "uv run", "ML course"):
        assert forbidden not in text
    assert "context" in text.lower()
    assert "adapter" in text.lower()

def test_payload_accepts_non_ml_source_identifiers(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["meta"]["experience_id"] = "roman-architecture"
    payload["meta"]["sources"] = [
        "knowledge/architecture.md",
        "https://example.edu/reference",
        "kb:history/roman-architecture",
    ]
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge/architecture.md").write_text("Source", encoding="utf-8")
    assert validate_payload(payload, repository_root=tmp_path) == []
```

- [ ] **Step 2: Run tests and confirm they fail because the core does not exist**

Run: `uv run pytest tests/test_interactive_learning_experience_builder_skill.py tests/test_lecture_site_generator.py -q`

Expected: FAIL on missing core files and unsupported generic metadata/sources.

- [ ] **Step 3: Move the reusable implementation and generalize metadata/source validation**

Implement `meta.experience_id`, accept safe repository-relative paths, `http://`
or `https://` provenance URLs, and explicit identifiers matching
`^[A-Za-z][A-Za-z0-9+.-]*:[^/].+`. Only repository-relative paths are checked
for existence. Remove hard-coded `lectures`, `okf`, private-name, and
repository-parent assumptions from the core.

- [ ] **Step 4: Write the discovery and adapter workflow**

The core `SKILL.md` must require reading repository instructions, inventorying
knowledge sources and validation/publishing conventions, writing an experience
specification, and creating a thin adapter only for recurring workflows. The
adapter template must contain placeholders for stable local constraints but no
hard-coded domain paths.

- [ ] **Step 5: Run the focused tests**

Run: `uv run pytest tests/test_interactive_learning_experience_builder_skill.py tests/test_lecture_site_generator.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add .agents/skills/interactive-learning-experience-builder tests/test_interactive_learning_experience_builder_skill.py tests/test_lecture_site_generator.py
git commit -m "feat: add portable learning experience builder"
```

### Task 2: Convert the ML-course skill into an adapter

**Files:**
- Modify: `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md`
- Modify: `.agents/skills/ml-course-interactive-learning-assistant/agents/openai.yaml`
- Delete: `.agents/skills/ml-course-interactive-learning-assistant/references/content-contract.md`
- Modify: `.agents/skills/ml-course-textbook-contributor/SKILL.md`
- Modify: `tests/test_interactive_learning_assistant_skill.py`
- Modify: `tests/test_interactive_learning_assistant_docs.py`

**Interfaces:**
- Consumes: the portable core workflow and the current ML-course repository.
- Produces: course source selection, lecture lookup, OKF read-only policy, public-safety rules, and course commands.

- [ ] **Step 1: Write failing adapter-parity tests**

```python
def test_adapter_preserves_course_constraints() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    for required in (
        "lectures/index.yaml",
        "lecture_notes.md",
        "okf/",
        "Do not modify `okf/`",
        "exactly 10",
        "interactive-learning-experience-builder",
    ):
        assert required in text

def test_textbook_skill_requires_mobile_quiz_contract() -> None:
    text = Path(".agents/skills/ml-course-textbook-contributor/SKILL.md").read_text()
    assert "wrong answers" in text.lower()
    assert "sticky progress" in text.lower()
    assert "mobile chrome" in text.lower()
```

- [ ] **Step 2: Run the adapter tests and confirm failure**

Run: `uv run pytest tests/test_interactive_learning_assistant_skill.py tests/test_interactive_learning_assistant_docs.py -q`

Expected: FAIL because the adapter still owns the reusable implementation and
the textbook skill lacks the mobile contract.

- [ ] **Step 3: Rewrite the course adapter and update textbook rules**

Point generation and validation commands to the core scripts/template. Keep the
four learner settings, three ten-question banks, course source hierarchy,
read-only OKF rule, private-source exclusion, `file://` verification, and
course-specific test/deploy commands.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_interactive_learning_assistant_skill.py tests/test_interactive_learning_assistant_docs.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add .agents/skills/ml-course-interactive-learning-assistant .agents/skills/ml-course-textbook-contributor tests/test_interactive_learning_assistant_skill.py tests/test_interactive_learning_assistant_docs.py
git commit -m "refactor: make ML learning assistant a repository adapter"
```

### Task 3: Fix the reusable quiz state machine

**Files:**
- Modify: `.agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html`
- Modify: `tests/test_lecture_site_generator.py`

**Interfaces:**
- Consumes: the current question and selected values.
- Produces: per-question `{attempts, firstAttemptCorrect, complete}` and completed responses used by progress/results.

- [ ] **Step 1: Write failing template-contract tests**

```python
def test_quiz_retry_state_contract() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    for hook in (
        "firstAttemptCorrect",
        "attempts",
        "clearQuizFeedback",
        "disableQuestionInputs",
        "if (!correct)",
        "totalAttempts",
    ):
        assert hook in template
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `uv run pytest tests/test_lecture_site_generator.py -q`

Expected: FAIL because wrong and correct answers currently share the same
completion path.

- [ ] **Step 3: Implement the minimal state machine**

On selection/input, call `clearQuizFeedback()` unless the question is complete.
On wrong submission, increment attempts, record first-attempt correctness once,
show feedback, and keep Check visible. On correct submission, mark complete,
append the completed response, disable inputs, hide Check, and reveal Next.
Make results show `First try: X/10` and `Total attempts: Y`.

- [ ] **Step 4: Reset every quiz field on Retry**

Reset index, response list, per-question state, feedback/result visibility,
disabled inputs, progress, and Check/Next state while leaving learner settings
unchanged.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_lecture_site_generator.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html tests/test_lecture_site_generator.py
git commit -m "fix: keep quiz retries on the current question"
```

### Task 4: Add mobile sticky progress

**Files:**
- Modify: `.agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html`
- Modify: `.agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py`
- Modify: `tests/test_lecture_site_generator.py`

**Interfaces:**
- Produces: `.progress-panel` with sticky positioning, safe-area top offset, compact mobile layout, and focus-safe scroll padding.

- [ ] **Step 1: Write failing sticky-progress tests**

```python
def test_template_has_mobile_sticky_progress_contract() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    for hook in (
        'class="panel progress-panel"',
        "position: sticky",
        "env(safe-area-inset-top",
        "scroll-padding-top",
        "@media (max-width:",
    ):
        assert hook in template
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `uv run pytest tests/test_lecture_site_generator.py -q`

Expected: FAIL because the progress panel is static.

- [ ] **Step 3: Implement sticky and compact mobile styles**

Add a non-transparent surface, border/shadow separation, safe-area-aware top
offset, `z-index`, compact mobile spacing, and scroll padding/margins that keep
focused quiz content visible.

- [ ] **Step 4: Require the sticky contract in the validator**

Extend `validate_html()` to report a missing progress panel or missing sticky
style without requiring a particular color or visual theme.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/test_lecture_site_generator.py -q`

Expected: PASS.

```powershell
git add .agents/skills/interactive-learning-experience-builder tests/test_lecture_site_generator.py
git commit -m "fix: keep learning progress visible on mobile"
```

### Task 5: Prove portability and regenerate the EDA experience

**Files:**
- Modify: `lecture_experiences/content/lecture_01_eda.json`
- Regenerate: `lecture_experiences/lecture_01_eda/index.html`
- Modify: `tests/test_eda_lecture_experience.py`
- Create: `tests/test_learning_experience_portability.py`
- Modify: `docs/interactive-lecture-learning-assistant.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/evals/interactive-learning-assistant-baseline.md`

**Interfaces:**
- Consumes: core generator/validator and ML-course adapter.
- Produces: a working EDA demo plus an unrelated-repository smoke test.

- [ ] **Step 1: Write a failing unrelated-repository smoke test**

Create a temporary `knowledge/history.md`, a valid non-ML payload using
`meta.experience_id`, copy the core skill into the temporary repository,
generate `site/index.html`, and assert that validation succeeds without
`AGENTS.md`, `uv`, or publishing configuration.

- [ ] **Step 2: Run the smoke and EDA tests and confirm failure**

Run: `uv run pytest tests/test_learning_experience_portability.py tests/test_eda_lecture_experience.py -q`

Expected: FAIL until paths and EDA metadata use the portable core.

- [ ] **Step 3: Update EDA metadata and regenerate deterministically**

Run:

```powershell
uv run python .agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py --content lecture_experiences/content/lecture_01_eda.json --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html --output lecture_experiences/lecture_01_eda/index.html
uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py lecture_experiences/lecture_01_eda/index.html
```

Expected: generated file written; validator prints success and exits 0.

- [ ] **Step 4: Update documentation and skill evaluation**

Document the portable-core/adapter split, context profile, optional adapter
creation, canonical/global synchronization, and the corrected retry semantics.
Record the old skill's hard-coded-path and mobile-state failures in the baseline
evaluation.

- [ ] **Step 5: Run focused and full automated checks**

Run: `uv run pytest tests/test_learning_experience_portability.py tests/test_eda_lecture_experience.py tests/test_interactive_learning_assistant_skill.py tests/test_interactive_learning_experience_builder_skill.py tests/test_lecture_site_generator.py -q`

Expected: PASS.

Run: `uv run pytest`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add lecture_experiences .agents/skills docs README.md AGENTS.md tests
git commit -m "feat: publish portable learning experience workflow"
```

### Task 6: Verify Android behavior, install globally, and synchronize repositories

**Files:**
- Install/update: `C:/Users/AndrD/.codex/skills/interactive-learning-experience-builder/`
- Update repository branches for both configured remotes.

**Interfaces:**
- Consumes: canonical core skill and generated EDA HTML.
- Produces: verified Android-like Chrome behavior, byte-matching global skill, and synchronized main branches.

- [ ] **Step 1: Run Android-like browser regression**

Use Chrome device emulation with a 390×844 viewport, touch, Android user agent,
and device pixel ratio when supported. Verify:

1. Challenge mode.
2. Wrong answer shows feedback and does not reveal Next.
3. Selecting a different answer clears feedback.
4. A second wrong answer stays on the question.
5. A correct answer disables inputs and reveals Next.
6. Next advances exactly once.
7. Sticky progress remains visible without covering focused controls.
8. Retry resets the entire quiz but keeps settings.

- [ ] **Step 2: Run complete repository verification**

Run:

```powershell
uv run ruff format --check .agents/skills/interactive-learning-experience-builder/scripts tests
uv run ruff check .agents/skills/interactive-learning-experience-builder/scripts tests
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
```

Expected: every command exits 0.

- [ ] **Step 3: Synchronize the canonical core into the global skills directory**

Use the skill-installer workflow to install/update
`interactive-learning-experience-builder`. Compare relative file lists and
SHA-256 hashes for `SKILL.md`, `agents/`, `assets/`, `references/`, and
`scripts/`; expected result is no drift.

- [ ] **Step 4: Merge and push both main branches**

Merge the verified feature commits into the teacher and student `main`
branches without discarding repository-specific documentation. Push each
configured remote and verify both remote tips.

- [ ] **Step 5: Verify CI and the deployed demo**

Wait for GitHub Actions on both repositories. Confirm the student Pages demo
returns HTTP 200 and rerun the mobile quiz regression against the deployed URL.

