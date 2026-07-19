# Build Week Learning Companion Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement the assigned task. Tasks 1 and 2
> run concurrently in isolated worktrees. Task 3 runs only after both branches
> are complete. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add grounded Regression and Classification learning companions that
prove the portable skill is reusable, then integrate and verify the
three-companion Build Week showcase.

**Architecture:** Keep the portable core and ML-course adapter unchanged.
Each parallel task owns one payload, one generated HTML artifact, and one
topic-specific regression test. A sequential integration task updates shared
discovery documentation, verifies all published demo routes, performs visual
review, and records evidence for the later human-led submission phase.

**Tech Stack:** Python 3, JSON, deterministic single-file HTML, the existing
portable generator and validator, pytest, Ruff, the repository preview builder,
Git worktrees, and PowerShell.

## Global Constraints

- Commit this plan on `basics`, then create all worktrees from that unchanged
  `basics` tip. Create both parallel worktrees before launching either task.
- Do not run Tasks 1 and 2 in the same checkout.
- Do not modify the portable core or the ML-course adapter.
- Do not add visualization types, runtime dependencies, or repository adapters.
- Use only canonical public sources belonging to the selected lecture plus
  permitted read-only `okf/` sources.
- Do not read or cite private solutions, teacher materials, answer keys,
  grading data, unpublished drafts, or untracked quiz workbooks.
- Tasks 1 and 2 preserve `okf/` byte-for-byte. Task 3 may modify only
  `okf/index.md` to explain the textbook's fast-review discovery surface.
- Preserve unrelated untracked `catboost_info/` and `quizzes/` directories.
- Generate HTML through
  `.agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py`;
  never edit generated HTML by hand.
- Use only `histogram`, `boxplot`, `scatter`, and `missingness`
  visualizations.
- Every payload contains `foundations`, `applied`, and `challenge` quiz banks
  with exactly ten questions per bank.
- Every visualization has a readable fallback conveying the same essential
  lesson without JavaScript or color.
- Every page remains self-contained and opens through `file://` without an
  account, server, CDN, external font, or network request.
- When `uv` is unavailable, use the existing environment at
  `C:\projects\personal\ml-course\.venv\Scripts\python.exe`; isolated
  worktrees do not contain the ignored `.venv` directory.
- Each task commits only the files it owns.

## Execution Topology

Create the parallel worktrees from the repository parent directory:

```powershell
git -C C:\projects\personal\ml-course worktree add `
  C:\projects\personal\ml-course-regression `
  -b codex/build-week-regression basics

git -C C:\projects\personal\ml-course worktree add `
  C:\projects\personal\ml-course-classification `
  -b codex/build-week-classification basics
```

Run Task 1 in `C:\projects\personal\ml-course-regression` and Task 2 in
`C:\projects\personal\ml-course-classification` at the same time.

After both branches contain a successful task commit, create the integration
worktree:

```powershell
git -C C:\projects\personal\ml-course worktree add `
  C:\projects\personal\ml-course-build-week-integration `
  -b codex/build-week-showcase basics
```

Run Task 3 only in
`C:\projects\personal\ml-course-build-week-integration`.

---

### Task 1: Regression Learning Companion

**Files:**

- Create:
  `lecture_experiences/content/lecture_04_regression.json`
- Create:
  `lecture_experiences/lecture_04_regression/index.html`
- Create:
  `tests/test_regression_lecture_experience.py`

**Interfaces:**

- Consumes:
  `generate_course_site(content_path, template_path, output_path,
  lecture_slug, repository_root) -> Path` from the existing course wrapper.
- Consumes:
  `validate_html(path: Path) -> list[str]` from the existing offline
  validator.
- Produces:
  a source-grounded payload with experience ID `lecture-04-regression`, a
  deterministic HTML artifact, and a focused regression test.

- [ ] **Step 1: Verify the isolated baseline**

Run:

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse basics
git status --short
```

Expected:

```text
codex/build-week-regression
The HEAD and basics SHAs are identical.
```

The status output must not include edits outside this task's three owned files.

- [ ] **Step 2: Read the required workflow and sources**

Read completely:

```text
AGENTS.md
docs/superpowers/specs/2026-07-19-build-week-learning-companion-showcase-design.md
.agents/skills/interactive-learning-experience-builder/SKILL.md
.agents/skills/interactive-learning-experience-builder/references/context-discovery.md
.agents/skills/interactive-learning-experience-builder/references/content-contract.md
.agents/skills/ml-course-interactive-learning-assistant/SKILL.md
lecture_experiences/content/lecture_01_eda.json
tests/test_eda_lecture_experience.py
lectures/index.yaml
lectures/lecture_04_regression/README.md
lectures/lecture_04_regression/lecture_notes.md
lectures/lecture_04_regression/links.yaml
lectures/lecture_04_regression/lecture_examples/README.md
lectures/lecture_04_regression/practical_session/README.md
```

Do not use the slides, private sources, untracked workbooks, or material from a
different lecture as payload provenance.

- [ ] **Step 3: Write the failing regression contract test**

Create `tests/test_regression_lecture_experience.py` with the same loading,
OKF-hash, offline-validation, and deterministic-regeneration pattern used by
`tests/test_eda_lecture_experience.py`.

The test must define these exact constants:

```python
CONTENT_PATH = ROOT / "lecture_experiences" / "content" / "lecture_04_regression.json"
SITE_PATH = ROOT / "lecture_experiences" / "lecture_04_regression" / "index.html"
LECTURE_SLUG = "lecture_04_regression"
EXPERIENCE_ID = "lecture-04-regression"
LEVELS = ("foundations", "applied", "challenge")
EXPECTED_CONCEPTS = {
    "regression-problem-types",
    "ols-fitted-values-residuals",
    "assumptions-as-diagnostics",
    "multicollinearity-instability",
    "ridge-lasso-regularization",
    "scaling-for-regularization",
    "regression-metrics",
    "validation-overfitting-interpretation",
}
EXPECTED_VISUALIZATION_TYPES = {"scatter", "histogram", "boxplot"}
```

The test must assert:

```python
assert payload["meta"]["experience_id"] == EXPERIENCE_ID
assert {concept["id"] for concept in payload["concepts"]} == EXPECTED_CONCEPTS
assert {item["type"] for item in payload["visualizations"]} == EXPECTED_VISUALIZATION_TYPES
assert all(len(payload["quizzes"][level]) == 10 for level in LEVELS)
assert len({question["id"] for question in questions}) == 30
assert all(question["answer"] for question in questions)
assert all(question["explanation"].strip() for question in questions)
assert all(question["concept"] in EXPECTED_CONCEPTS for question in questions)
assert all(
    source.startswith("lectures/lecture_04_regression/")
    or source.startswith("okf/")
    for source in payload["meta"]["sources"]
)
```

For single-choice and multiple-choice questions, assert that every answer uses
only declared options. Assert that all concept sources use the same lecture or
`okf/` allowlist. Assert that the committed artifact contains no unexpanded
template tokens, passes `validate_html`, and is byte-for-byte identical to a
temporary artifact regenerated through `generate_course_site`. Hash every file
under `okf/` before generation and assert that the hashes are unchanged
afterward.

- [ ] **Step 4: Run the test and verify the RED state**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest `
  tests/test_regression_lecture_experience.py -q
```

Expected: failure because
`lecture_experiences/content/lecture_04_regression.json` does not exist.

- [ ] **Step 5: Author the grounded Regression payload**

Create
`lecture_experiences/content/lecture_04_regression.json` with exactly the six
top-level keys required by the portable content contract:

```text
meta, defaults, concepts, visualizations, quizzes, break_prompts
```

Use these defaults:

```json
{
  "difficulty": "foundations",
  "focus_mode": true,
  "color_blind": true,
  "break_prompts": true
}
```

Use the eight concept IDs from Step 3 and this concept arc:

1. regression targets and model families;
2. fitted values, residuals, and ordinary least squares;
3. assumptions as diagnostic questions;
4. multicollinearity and coefficient instability;
5. Ridge and Lasso regularization;
6. why scaling matters for regularization;
7. MAE, MSE, RMSE, and R-squared trade-offs; and
8. validation, overfitting, and interpretation limits.

Use these visualization identities and schemas:

```text
reg-fitted-relationship       scatter
reg-residual-distribution     histogram
reg-residual-outliers         boxplot
```

Embed small illustrative numeric data directly in the payload. Explain that the
data is illustrative. Do not imply that it is copied from an external dataset.
The scatter must make its trend-line control meaningful. The histogram must
provide at least four valid bin choices. The box plot must provide fence
multipliers `1`, `1.5`, and `2` with data that produces visibly different
flagging behavior.

Use these exact question ID prefixes:

```text
reg-f-01 through reg-f-10
reg-a-01 through reg-a-10
reg-c-01 through reg-c-10
```

Every explanation, interpretation, misconception, quiz answer, and corrective
feedback item must be supported by the named Lecture 04 sources.

- [ ] **Step 6: Generate and validate the Regression artifact**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py `
  --lecture-slug lecture_04_regression `
  --content lecture_experiences/content/lecture_04_regression.json `
  --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
  --output lecture_experiences/lecture_04_regression/index.html

& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  lecture_experiences/lecture_04_regression/index.html
```

Expected:

```text
GENERATED: lecture_experiences\lecture_04_regression\index.html
VALID: lecture_experiences\lecture_04_regression\index.html
```

- [ ] **Step 7: Run focused verification**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest `
  tests/test_regression_lecture_experience.py `
  tests/test_eda_lecture_experience.py `
  tests/test_interactive_learning_assistant_skill.py `
  tests/test_interactive_learning_experience_builder_skill.py `
  tests/test_learning_experience_portability.py -q
```

Expected: all collected tests pass, except documented environment-dependent
skips.

Regenerate the Regression artifact once more and confirm:

```powershell
git diff --exit-code -- `
  lecture_experiences/lecture_04_regression/index.html
```

Expected: no diff.

- [ ] **Step 8: Commit only the Regression task**

Run:

```powershell
git add -- `
  lecture_experiences/content/lecture_04_regression.json `
  lecture_experiences/lecture_04_regression/index.html `
  tests/test_regression_lecture_experience.py

git diff --cached --check
git commit -m "feat: add regression learning companion"
git status --short
```

Expected: the commit contains exactly three files. Return the commit SHA, source
inventory, verification output, and any documented skips.

---

### Task 2: Classification Part 1 Learning Companion

**Files:**

- Create:
  `lecture_experiences/content/lecture_05_classification_part_1.json`
- Create:
  `lecture_experiences/lecture_05_classification_part_1/index.html`
- Create:
  `tests/test_classification_part_1_lecture_experience.py`

**Interfaces:**

- Consumes:
  `generate_course_site(content_path, template_path, output_path,
  lecture_slug, repository_root) -> Path` from the existing course wrapper.
- Consumes:
  `validate_html(path: Path) -> list[str]` from the existing offline
  validator.
- Produces:
  a source-grounded payload with experience ID
  `lecture-05-classification-part-1`, a deterministic HTML artifact, and a
  focused classification test.

- [ ] **Step 1: Verify the isolated baseline**

Run:

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse basics
git status --short
```

Expected:

```text
codex/build-week-classification
The HEAD and basics SHAs are identical.
```

The status output must not include edits outside this task's three owned files.

- [ ] **Step 2: Read the required workflow and sources**

Read completely:

```text
AGENTS.md
docs/superpowers/specs/2026-07-19-build-week-learning-companion-showcase-design.md
.agents/skills/interactive-learning-experience-builder/SKILL.md
.agents/skills/interactive-learning-experience-builder/references/context-discovery.md
.agents/skills/interactive-learning-experience-builder/references/content-contract.md
.agents/skills/ml-course-interactive-learning-assistant/SKILL.md
lecture_experiences/content/lecture_01_eda.json
tests/test_eda_lecture_experience.py
lectures/index.yaml
lectures/lecture_05_classification_part_1/README.md
lectures/lecture_05_classification_part_1/lecture_notes.md
lectures/lecture_05_classification_part_1/links.yaml
lectures/lecture_05_classification_part_1/lecture_examples/README.md
lectures/lecture_05_classification_part_1/practical_session/README.md
```

Do not use the slides, private sources, untracked workbooks, or material from a
different lecture as payload provenance.

- [ ] **Step 3: Write the failing Classification contract test**

Create `tests/test_classification_part_1_lecture_experience.py` with the same
loading, OKF-hash, offline-validation, and deterministic-regeneration pattern
used by `tests/test_eda_lecture_experience.py`.

The test must define these exact constants:

```python
CONTENT_PATH = (
    ROOT
    / "lecture_experiences"
    / "content"
    / "lecture_05_classification_part_1.json"
)
SITE_PATH = (
    ROOT
    / "lecture_experiences"
    / "lecture_05_classification_part_1"
    / "index.html"
)
LECTURE_SLUG = "lecture_05_classification_part_1"
EXPERIENCE_ID = "lecture-05-classification-part-1"
LEVELS = ("foundations", "applied", "challenge")
EXPECTED_CONCEPTS = {
    "classification-problem-types",
    "knn-distance-scaling",
    "decision-tree-impurity",
    "tree-overfitting-control",
    "logistic-probabilities-thresholds",
    "confusion-matrix-outcomes",
    "precision-recall-fscore",
    "roc-auc-log-loss",
}
EXPECTED_VISUALIZATION_TYPES = {"histogram", "scatter", "boxplot"}
```

The test must assert:

```python
assert payload["meta"]["experience_id"] == EXPERIENCE_ID
assert {concept["id"] for concept in payload["concepts"]} == EXPECTED_CONCEPTS
assert {item["type"] for item in payload["visualizations"]} == EXPECTED_VISUALIZATION_TYPES
assert all(len(payload["quizzes"][level]) == 10 for level in LEVELS)
assert len({question["id"] for question in questions}) == 30
assert all(question["answer"] for question in questions)
assert all(question["explanation"].strip() for question in questions)
assert all(question["concept"] in EXPECTED_CONCEPTS for question in questions)
assert all(
    source.startswith("lectures/lecture_05_classification_part_1/")
    or source.startswith("okf/")
    for source in payload["meta"]["sources"]
)
```

For single-choice and multiple-choice questions, assert that every answer uses
only declared options. Assert that all concept sources use the same lecture or
`okf/` allowlist. Assert that the committed artifact contains no unexpanded
template tokens, passes `validate_html`, and is byte-for-byte identical to a
temporary artifact regenerated through `generate_course_site`. Hash every file
under `okf/` before generation and assert that the hashes are unchanged
afterward.

- [ ] **Step 4: Run the test and verify the RED state**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest `
  tests/test_classification_part_1_lecture_experience.py -q
```

Expected: failure because
`lecture_experiences/content/lecture_05_classification_part_1.json` does not
exist.

- [ ] **Step 5: Author the grounded Classification payload**

Create
`lecture_experiences/content/lecture_05_classification_part_1.json` with
exactly the six top-level keys required by the portable content contract:

```text
meta, defaults, concepts, visualizations, quizzes, break_prompts
```

Use these defaults:

```json
{
  "difficulty": "foundations",
  "focus_mode": true,
  "color_blind": true,
  "break_prompts": true
}
```

Use the eight concept IDs from Step 3 and this concept arc:

1. binary, multiclass, multilabel, and ordinal targets;
2. KNN distance, scaling, and the role of `k`;
3. tree splits and impurity;
4. depth, pruning controls, and tree overfitting;
5. logistic scores, probabilities, and decision thresholds;
6. confusion-matrix outcomes;
7. precision, recall, F-scores, and class imbalance; and
8. ROC/AUC, log loss, and probability-sensitive evaluation.

Use these visualization identities and schemas:

```text
cls-score-distribution       histogram
cls-feature-separation       scatter
cls-margin-outliers          boxplot
```

Embed small illustrative numeric data directly in the payload. Explain that the
data is illustrative. Do not imply that it is copied from an external dataset.
The histogram must support discussion of score distribution without pretending
to be a threshold-control widget. The scatter must make its trend-line control
pedagogically defensible. The box plot must use margin-like or error-like
measurements and must not imply that every flagged value is a data error.

Use these exact question ID prefixes:

```text
cls-f-01 through cls-f-10
cls-a-01 through cls-a-10
cls-c-01 through cls-c-10
```

Every explanation, interpretation, misconception, quiz answer, and corrective
feedback item must be supported by the named Lecture 05 sources.

- [ ] **Step 6: Generate and validate the Classification artifact**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py `
  --lecture-slug lecture_05_classification_part_1 `
  --content lecture_experiences/content/lecture_05_classification_part_1.json `
  --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
  --output lecture_experiences/lecture_05_classification_part_1/index.html

& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  lecture_experiences/lecture_05_classification_part_1/index.html
```

Expected:

```text
GENERATED: lecture_experiences\lecture_05_classification_part_1\index.html
VALID: lecture_experiences\lecture_05_classification_part_1\index.html
```

- [ ] **Step 7: Run focused verification**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest `
  tests/test_classification_part_1_lecture_experience.py `
  tests/test_eda_lecture_experience.py `
  tests/test_interactive_learning_assistant_skill.py `
  tests/test_interactive_learning_experience_builder_skill.py `
  tests/test_learning_experience_portability.py -q
```

Expected: all collected tests pass, except documented environment-dependent
skips.

Regenerate the Classification artifact once more and confirm:

```powershell
git diff --exit-code -- `
  lecture_experiences/lecture_05_classification_part_1/index.html
```

Expected: no diff.

- [ ] **Step 8: Commit only the Classification task**

Run:

```powershell
git add -- `
  lecture_experiences/content/lecture_05_classification_part_1.json `
  lecture_experiences/lecture_05_classification_part_1/index.html `
  tests/test_classification_part_1_lecture_experience.py

git diff --cached --check
git commit -m "feat: add classification learning companion"
git status --short
```

Expected: the commit contains exactly three files. Return the commit SHA, source
inventory, verification output, and any documented skips.

---

### Task 3: Integrate, Publish, and Record Verification Evidence

**Files:**

- Modify: `README.md`
- Modify: `docs/interactive-lecture-learning-assistant.md`
- Modify: `lectures/lecture_04_regression/README.md`
- Modify: `lectures/lecture_05_classification_part_1/README.md`
- Modify: `okf/index.md`
- Modify: `tools/build_textbook_preview.py`
- Modify: `site/assets/textbook.css`
- Modify: `tests/test_interactive_learning_assistant_docs.py`
- Modify: `tests/test_textbook_preview.py`
- Create: `docs/build-week-integration-evidence.md`

**Interfaces:**

- Consumes:
  the branch tips of `codex/build-week-regression` and
  `codex/build-week-classification`.
- Consumes:
  committed artifacts under `lecture_experiences/*/index.html`.
- Produces:
  `LearningExperience(slug, experience_id, title, artifact, output)` build-time
  descriptors loaded from matching payload/artifact pairs.
- Produces:
  `_load_learning_experiences(experiences: Path, build_dir: Path) ->
  list[LearningExperience]`.
- Produces:
  `_learning_experience_cards_html(page: Page, experiences:
  list[LearningExperience]) -> str`.
- Produces:
  `_navigation_html(pages: list[Page], current: Page, experiences:
  list[LearningExperience]) -> str`.
- Produces:
  three discoverable offline/live demo routes, a fast-review section on the
  textbook homepage, fast-review links in every textbook sidebar, cleaned
  repository/authoring guidance, integration tests, and a verification
  evidence record for the later human-led submission phase.

- [ ] **Step 1: Verify the integration baseline and import both task commits**

Run:

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse basics
git status --short

$regressionCommit = git rev-parse codex/build-week-regression
$classificationCommit = git rev-parse codex/build-week-classification

git cherry-pick $regressionCommit
git cherry-pick $classificationCommit
```

Expected branch and base:

```text
codex/build-week-showcase
The HEAD and basics SHAs are identical before the two cherry-picks.
```

Confirm each imported commit changes exactly its three owned files:

```powershell
git show --stat --oneline $regressionCommit
git show --stat --oneline $classificationCommit
```

- [ ] **Step 2: Review both contributions before shared edits**

For each new payload and artifact:

1. read the named sources and spot-check every concept;
2. confirm all 30 quiz questions have one unambiguous grounded answer;
3. confirm illustrative data is labeled as illustrative;
4. confirm visualization controls are meaningful;
5. confirm the generated artifact was not hand-edited;
6. run its topic-specific test and offline validator; and
7. confirm `okf/` is unchanged from the common `basics` base.

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest `
  tests/test_regression_lecture_experience.py `
  tests/test_classification_part_1_lecture_experience.py -q

& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  lecture_experiences/lecture_04_regression/index.html

& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  lecture_experiences/lecture_05_classification_part_1/index.html

git diff --exit-code basics -- okf
```

Expected: both tests pass, both artifacts report `VALID`, and the OKF diff is
empty.

- [ ] **Step 3: Write failing documentation and textbook-discovery tests**

In `tests/test_interactive_learning_assistant_docs.py`, replace the single EDA
constants with:

```python
DEMOS = {
    "lecture_01_eda": {
        "live": (
            "https://derandr.github.io/machine_learning_course_basics/"
            "demos/lecture_01_eda/"
        ),
        "offline": "lecture_experiences/lecture_01_eda/index.html",
        "lecture_readme": "lectures/lecture_01_eda/README.md",
    },
    "lecture_04_regression": {
        "live": (
            "https://derandr.github.io/machine_learning_course_basics/"
            "demos/lecture_04_regression/"
        ),
        "offline": "lecture_experiences/lecture_04_regression/index.html",
        "lecture_readme": "lectures/lecture_04_regression/README.md",
    },
    "lecture_05_classification_part_1": {
        "live": (
            "https://derandr.github.io/machine_learning_course_basics/"
            "demos/lecture_05_classification_part_1/"
        ),
        "offline": (
            "lecture_experiences/lecture_05_classification_part_1/index.html"
        ),
        "lecture_readme": (
            "lectures/lecture_05_classification_part_1/README.md"
        ),
    },
}
```

Update `test_learning_assistant_documentation_is_discoverable` to assert:

```python
readme = Path("README.md").read_text(encoding="utf-8")
guide_text = Path("docs/interactive-lecture-learning-assistant.md").read_text(
    encoding="utf-8"
)
for demo in DEMOS.values():
    assert demo["live"] in readme
    assert demo["offline"] in readme
    assert demo["live"] in guide_text
    assert demo["offline"] in guide_text
    lecture_text = Path(demo["lecture_readme"]).read_text(encoding="utf-8")
    assert demo["live"] in lecture_text
    assert demo["offline"] in lecture_text
```

Add exact assertions for the cleaned README structure:

```python
for heading in (
    "## Choose how to learn",
    "## Fast interactive reviews",
    "## Course map",
    "## Create interactive learning materials",
    "## Local setup",
    "## Contributing and reference",
    "## License",
):
    assert heading in readme
```

Add exact guide assertions:

```python
general_prompt = (
    "Use $interactive-learning-experience-builder to create a grounded, "
    "offline interactive learning experience from this repository's "
    "knowledge sources."
)
course_prompt = (
    "Use $ml-course-interactive-learning-assistant with "
    "$interactive-learning-experience-builder to create a grounded, "
    "accessible, self-contained review for a selected ML-course lecture."
)
for heading in (
    "## Choose the right skill",
    "## General-purpose workflow",
    "## ML-course workflow",
    "## Authoring and verification flow",
):
    assert heading in guide_text
assert general_prompt in guide_text
assert course_prompt in guide_text
```

Assert that `okf/index.md` contains `## Fast interactive reviews`, describes
the companions as complementary to the textbook, and contains no hardcoded
`demos/` URL.

In `tests/test_textbook_preview.py`, add `import pytest` and define:

```python
EXPECTED_DEMOS = {
    "lecture_01_eda": "Exploratory Data Analysis: Interactive Review",
    "lecture_04_regression": "Regression: Interactive Review",
    "lecture_05_classification_part_1": (
        "Classification Part 1: Interactive Review"
    ),
}
```

Replace the EDA-only publication test with one test that builds the preview and
asserts:

```python
home_html = (output / "index.html").read_text(encoding="utf-8")
assert '<section class="learning-experience-section">' in home_html
assert "<h2>Fast interactive reviews</h2>" in home_html
assert "<h2>Open a fast review</h2>" in home_html
assert '<section class="sidebar-section learning-experience-nav">' in home_html

for slug, expected_title in EXPECTED_DEMOS.items():
    source = Path("lecture_experiences") / slug / "index.html"
    published = output / "demos" / slug / "index.html"
    assert published.is_file()
    assert published.read_bytes() == source.read_bytes()
    assert expected_title in published.read_text(encoding="utf-8")
    assert expected_title in home_html
    assert f'demos/{slug}/index.html' in home_html

assert not (output / "demos" / "content").exists()
```

Load `okf-manifest.json`, collect `index.html` plus every `textbook_path`, and
assert that each rendered textbook page—not the copied demo pages—contains:

```python
assert '<section class="sidebar-section learning-experience-nav">' in html_text
assert "<h2>Fast reviews</h2>" in html_text
for expected_title in EXPECTED_DEMOS.values():
    assert expected_title in html_text
```

For
`supervised-learning/classification/classification-metrics.html`, assert the
sidebar uses these exact relative hrefs:

```python
for slug in EXPECTED_DEMOS:
    assert f'../../../demos/{slug}/index.html' in metrics_html
```

Add a failure-path test that creates a temporary experiences directory with
`lecture_orphan/index.html` but no matching content JSON and asserts:

```python
with pytest.raises(
    SystemExit,
    match=r"lecture_orphan.*matching payload",
):
    build_textbook_preview(
        output=tmp_path / "missing-payload-build",
        experiences=experiences,
    )
```

Then create `content/lecture_orphan.json` containing invalid JSON and assert a
second build raises `SystemExit` whose message includes both
`lecture_orphan.json` and `invalid JSON`.

- [ ] **Step 4: Run the new tests and verify the RED state**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest `
  tests/test_interactive_learning_assistant_docs.py `
  tests/test_textbook_preview.py -q
```

Expected: failures because the new demos are not documented, the builder has no
`LearningExperience` discovery model, and no textbook page renders fast-review
navigation.

- [ ] **Step 5: Implement data-driven companion discovery**

In `tools/build_textbook_preview.py`, add:

```python
@dataclass(frozen=True)
class LearningExperience:
    """One committed standalone learning companion."""

    slug: str
    experience_id: str
    title: str
    artifact: Path
    output: Path
```

Add:

```python
def _load_learning_experiences(
    experiences: Path,
    build_dir: Path,
) -> list[LearningExperience]:
    items: list[LearningExperience] = []
    if not experiences.is_dir():
        return items
    for artifact in sorted(experiences.glob("*/index.html")):
        slug = artifact.parent.name
        payload_path = experiences / "content" / f"{slug}.json"
        if not payload_path.is_file():
            raise SystemExit(
                f"learning experience {slug} has no matching payload: "
                f"{payload_path}"
            )
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"learning experience payload has invalid JSON: "
                f"{payload_path}: {error}"
            ) from error
        if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
            raise SystemExit(
                f"learning experience payload has no metadata object: "
                f"{payload_path}"
            )
        meta = payload["meta"]
        experience_id = meta.get("experience_id")
        title = meta.get("title")
        if not isinstance(experience_id, str) or not experience_id.strip():
            raise SystemExit(
                f"learning experience payload has no experience_id: "
                f"{payload_path}"
            )
        if not isinstance(title, str) or not title.strip():
            raise SystemExit(
                f"learning experience payload has no title: {payload_path}"
            )
        items.append(
            LearningExperience(
                slug=slug,
                experience_id=experience_id,
                title=title,
                artifact=artifact,
                output=build_dir / "demos" / slug / "index.html",
            )
        )
    return items
```

Keep `_copy_lecture_experiences` responsible only for byte-for-byte copying.
Do not make it parse content or render navigation.

- [ ] **Step 6: Render homepage cards and sidebar links**

Add:

```python
def _learning_experience_cards_html(
    page: Page,
    experiences: list[LearningExperience],
) -> str:
    if page.okf_path != "/index.md" or not experiences:
        return ""
    cards = []
    for experience in experiences:
        href = _relative_href(page.output, experience.output)
        cards.append(
            '<a class="relationship-card" '
            f'href="{html.escape(href, quote=True)}">'
            "<span>Fast interactive review</span>"
            f"<strong>{html.escape(experience.title)}</strong>"
            "<small>Focused explanations, interactive exploration, "
            "and corrective quizzes.</small></a>"
        )
    return (
        '<section class="learning-experience-section">'
        "<h2>Open a fast review</h2>"
        f'<div class="relationship-grid">{"".join(cards)}</div></section>'
    )
```

Change `_navigation_html` to accept `experiences` and return one `<nav>` with
two explicit sections:

```python
def _navigation_html(
    pages: list[Page],
    current: Page,
    experiences: list[LearningExperience],
) -> str:
```

The first section must use:

```html
<section class="sidebar-section">
  <h2>Textbook pages</h2>
  <ul>...</ul>
</section>
```

The second section must use:

```html
<section class="sidebar-section learning-experience-nav">
  <h2>Fast reviews</h2>
  <ul>...</ul>
</section>
```

Generate each companion href with
`_relative_href(current.output, experience.output)`.

Change `_render_page` to accept `experiences`, pass it to `_navigation_html`,
and insert `_learning_experience_cards_html(page, experiences)` immediately
after the rendered Markdown body.

In `build_textbook_preview`, load descriptors before copying artifacts:

```python
learning_experiences = _load_learning_experiences(
    experiences,
    build_dir.resolve(),
)
_copy_lecture_experiences(experiences, build_dir)
```

Pass `learning_experiences` into every `_render_page` call.

- [ ] **Step 7: Add textbook copy and styling**

Update `okf/index.md` with this durable distinction, using original prose
consistent with the repository voice:

```markdown
## Interactive textbook

Use the textbook for concise concepts, connected learning paths, source links,
and browser labs.

## Fast interactive reviews

Use fast interactive reviews for a short explanation-exploration-quiz-feedback
loop focused on one lecture. They complement the textbook and full lecture
materials; they do not replace either one. The rendered textbook discovers
verified companion artifacts automatically.
```

Do not add hardcoded `demos/` links to `okf/index.md`.

In `site/assets/textbook.css`:

- include `.learning-experience-section` in the existing bordered-section rule;
- add `.sidebar-section + .sidebar-section { margin-top: 1.25rem; }`;
- add `.learning-experience-nav { border-top: 1px solid var(--line);
  padding-top: 1rem; }`;
- add a visible `:focus-visible` outline for `.sidebar a` and
  `.relationship-card`; and
- preserve the existing one-column mobile layout and static sidebar behavior.

- [ ] **Step 8: Refactor README and authoring guidance**

Refactor `README.md` into this order:

```markdown
# Machine Learning Course
## Choose how to learn
## Fast interactive reviews
## Course map
## Create interactive learning materials
## Local setup
## Contributing and reference
## Repository map
## License
```

Under `Choose how to learn`, provide four concise routes:

```text
Interactive textbook — connected concepts, learning paths, and browser labs.
Fast interactive reviews — focused offline-capable practice with quizzes.
Full lecture materials — notes, slides, examples, and practical sessions.
Hands-on notebooks — runnable demonstrations and classroom exercises.
```

Under `Fast interactive reviews`, include the live and offline paths for EDA,
Regression, and Classification Part 1. Preserve the complete lecture map.
Retain only the essential local commands:

```powershell
uv sync
uv run python tools/check_notebook_environment.py
uv run jupyter lab
```

Link detailed environment and optional dependency guidance to
`docs/student-quickstart.md`. Remove duplicated setup prose, malformed code
fences, and low-level repository explanations already owned by linked guides.
Do not add Build Week deadlines or judging language.

Under `Create interactive learning materials`, include a compact two-row table:

```text
General repository or knowledge base | interactive-learning-experience-builder
This ML course                         | ml-course-interactive-learning-assistant
                                        + interactive-learning-experience-builder
```

Link to `docs/interactive-lecture-learning-assistant.md` for the complete
workflow.

Extend `docs/interactive-lecture-learning-assistant.md` with these exact
headings:

```markdown
## Choose the right skill
## General-purpose workflow
## ML-course workflow
## Authoring and verification flow
```

Include the exact default prompts:

```text
Use $interactive-learning-experience-builder to create a grounded, offline
interactive learning experience from this repository's knowledge sources.

Use $ml-course-interactive-learning-assistant with
$interactive-learning-experience-builder to create a grounded, accessible,
self-contained review for a selected ML-course lecture.
```

Explain:

```text
sources -> context profile -> experience specification -> grounded JSON
payload -> deterministic generation -> offline validation -> browser review
-> optional publication
```

Keep the existing exact generation and validation commands. Explain that the
portable core owns reusable behavior, the adapter owns stable course policy,
and each lecture owns only its payload and generated artifact.

Add the live and offline links for all three companions to the guide and the
matching lecture READMEs.

- [ ] **Step 9: Run automated integration verification**

Run:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest `
  tests/test_eda_lecture_experience.py `
  tests/test_regression_lecture_experience.py `
  tests/test_classification_part_1_lecture_experience.py `
  tests/test_interactive_learning_assistant_skill.py `
  tests/test_interactive_learning_assistant_docs.py `
  tests/test_interactive_learning_experience_builder_skill.py `
  tests/test_learning_experience_portability.py `
  tests/test_textbook_preview.py -q

& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  tools/validate_okf.py okf/ --strict-warnings
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' `
  tools/build_textbook_preview.py
& 'C:\projects\personal\ml-course\.venv\Scripts\python.exe' -m pytest -q
```

Run the required formatting, lint, and type checks:

```powershell
& 'C:\projects\personal\ml-course\.venv\Scripts\ruff.exe' format --check `
  src/mlcourse/okf_validation.py `
  tools/validate_okf.py `
  tools/build_textbook_preview.py `
  tests/test_okf_validation.py `
  tests/test_textbook_preview.py `
  tests/test_smoke.py

& 'C:\projects\personal\ml-course\.venv\Scripts\ruff.exe' check `
  src/mlcourse/okf_validation.py `
  tools/validate_okf.py `
  tools/build_textbook_preview.py `
  tests/test_okf_validation.py `
  tests/test_textbook_preview.py `
  tests/test_smoke.py `
  tests/test_regression_lecture_experience.py `
  tests/test_classification_part_1_lecture_experience.py `
  tests/test_interactive_learning_assistant_docs.py

& 'C:\projects\personal\ml-course\.venv\Scripts\ty.exe' check `
  src/mlcourse/okf_validation.py `
  tools/validate_okf.py `
  tools/build_textbook_preview.py
```

Expected: all applicable checks pass. Record exact pass counts and every
documented skip.

- [ ] **Step 10: Perform desktop and mobile visual verification**

Open `site/_build/index.html` and at least one deeply nested textbook page.
Verify:

1. the homepage contains three readable fast-review cards;
2. every textbook sidebar contains the separate Fast reviews group;
3. all card and sidebar links open the correct copied demo;
4. keyboard focus is visible;
5. the combined sidebar remains usable at desktop and mobile widths; and
6. the textbook/fast-review distinction is understandable without repository
   context.

Open each canonical offline artifact directly:

```text
lecture_experiences/lecture_01_eda/index.html
lecture_experiences/lecture_04_regression/index.html
lecture_experiences/lecture_05_classification_part_1/index.html
```

For every artifact verify:

1. all concepts render and source labels are readable;
2. every visualization control changes its intended view;
3. fallback text communicates the essential lesson;
4. Foundations, Applied, and Challenge each contain ten questions;
5. wrong answer, changed answer, correct answer, Next, results, and whole-quiz
   Retry follow the documented state machine;
6. focus-friendly and color-blind-safe modes remain readable;
7. break prompts can be enabled and disabled;
8. keyboard focus is visible and logical;
9. reduced-motion mode is respected;
10. disabled storage does not break the experience; and
11. Android Chrome emulation shows no obscured progress, feedback, or controls.

Do not change the companion runtime during this task. Record content-specific
problems against the owning payload and stop if a shared-runtime defect would
require redesign.

- [ ] **Step 11: Create the integration evidence record**

Create `docs/build-week-integration-evidence.md` with these exact headings:

```markdown
# Build Week Learning Companion Integration Evidence

## Integrated commits
## Artifact inventory
## Source-policy verification
## Automated test evidence
## Deterministic regeneration
## Desktop visual review
## Mobile visual review
## Textbook discovery review
## README and authoring-guide review
## Published route expectations
## Remaining human-led submission work
```

Record exact commit SHAs, commands, pass counts, validator results, the three
offline paths, the three expected Pages routes, textbook card/sidebar findings,
README/guide findings, and concise companion visual findings.

Under `Remaining human-led submission work`, list only:

```text
- Verify deployed Pages after merge to main.
- Select the primary /feedback Codex session ID.
- Capture final screenshots.
- Record and upload the public video of less than three minutes.
- Write the Devpost description in the submitter's own voice.
- Complete the official rule and announcement checklist.
- Submit through Devpost before the deadline.
```

- [ ] **Step 12: Commit the integrated showcase**

Run:

```powershell
git add -- `
  README.md `
  docs/interactive-lecture-learning-assistant.md `
  docs/build-week-integration-evidence.md `
  lectures/lecture_04_regression/README.md `
  lectures/lecture_05_classification_part_1/README.md `
  okf/index.md `
  site/assets/textbook.css `
  tests/test_interactive_learning_assistant_docs.py `
  tests/test_textbook_preview.py `
  tools/build_textbook_preview.py

git diff --cached --check
git commit -m "feat: integrate Build Week learning companion showcase"
git status --short
```

Return the integration commit SHA, imported task SHAs, exact verification
results, visual-review findings, expected Pages routes, and remaining
human-led submission work.

---

## Copy-Paste Prompt A: Regression Agent

```text
Work only on the Regression learning companion in the isolated worktree
C:\projects\personal\ml-course-regression on branch
codex/build-week-regression.

Do not create or dispatch subagents. Use the existing approved design and plan:
- docs/superpowers/specs/2026-07-19-build-week-learning-companion-showcase-design.md
- docs/superpowers/plans/2026-07-19-build-week-learning-companion-showcase.md

Execute only "Task 1: Regression Learning Companion" from the plan. Follow
AGENTS.md and invoke the repository's interactive-learning-experience-builder,
ML-course interactive-learning assistant, executing-plans, TDD, and
verification skills as applicable.

Strict ownership:
- lecture_experiences/content/lecture_04_regression.json
- lecture_experiences/lecture_04_regression/index.html
- tests/test_regression_lecture_experience.py

Do not modify any other file. Do not modify the portable core, ML adapter, OKF,
shared docs, build scripts, configuration, catboost_info/, or quizzes/. Generate
the HTML only through the course wrapper. Use current visualization schemas
only. Include exactly three quiz banks with ten grounded questions each.

If uv is unavailable, use
C:\projects\personal\ml-course\.venv\Scripts\python.exe because the isolated
worktree has no local .venv. Complete the RED/GREEN test cycle, validate the
HTML, prove deterministic regeneration, run the focused suite, commit only the
three owned files with message
"feat: add regression learning companion", and return:
1. commit SHA;
2. exact files changed;
3. public source inventory;
4. test and validator results;
5. any skips or remaining concerns.

Do not start integration or submission work.
```

## Copy-Paste Prompt B: Classification Agent

```text
Work only on the Classification Part 1 learning companion in the isolated
worktree C:\projects\personal\ml-course-classification on branch
codex/build-week-classification.

Do not create or dispatch subagents. Use the existing approved design and plan:
- docs/superpowers/specs/2026-07-19-build-week-learning-companion-showcase-design.md
- docs/superpowers/plans/2026-07-19-build-week-learning-companion-showcase.md

Execute only "Task 2: Classification Part 1 Learning Companion" from the plan.
Follow AGENTS.md and invoke the repository's
interactive-learning-experience-builder, ML-course interactive-learning
assistant, executing-plans, TDD, and verification skills as applicable.

Strict ownership:
- lecture_experiences/content/lecture_05_classification_part_1.json
- lecture_experiences/lecture_05_classification_part_1/index.html
- tests/test_classification_part_1_lecture_experience.py

Do not modify any other file. Do not modify the portable core, ML adapter, OKF,
shared docs, build scripts, configuration, catboost_info/, or quizzes/. Generate
the HTML only through the course wrapper. Use current visualization schemas
only. Include exactly three quiz banks with ten grounded questions each.

If uv is unavailable, use
C:\projects\personal\ml-course\.venv\Scripts\python.exe because the isolated
worktree has no local .venv. Complete the RED/GREEN test cycle, validate the
HTML, prove deterministic regeneration, run the focused suite, commit only the
three owned files with message
"feat: add classification learning companion", and return:
1. commit SHA;
2. exact files changed;
3. public source inventory;
4. test and validator results;
5. any skips or remaining concerns.

Do not start integration or submission work.
```

## Copy-Paste Prompt C: Integration Agent

```text
Run this task only after branches codex/build-week-regression and
codex/build-week-classification have completed successfully.

Work in the isolated worktree
C:\projects\personal\ml-course-build-week-integration on branch
codex/build-week-showcase.

Do not create or dispatch subagents. Use the approved design and plan:
- docs/superpowers/specs/2026-07-19-build-week-learning-companion-showcase-design.md
- docs/superpowers/plans/2026-07-19-build-week-learning-companion-showcase.md

Execute only "Task 3: Integrate, Publish, and Record Verification Evidence"
from the plan. Follow AGENTS.md and use executing-plans, TDD,
ml-course-textbook-contributor, systematic verification, and browser-testing
skills as applicable.

Import the exact branch tips of codex/build-week-regression and
codex/build-week-classification. Confirm each imported commit changed only its
three owned files before proceeding.

Do not redesign or modify the portable core, ML adapter, visualization schemas,
configuration, catboost_info/, or quizzes/. The only allowed OKF edit is
okf/index.md. Review content grounding, implement data-driven textbook
discovery, add fast-review cards to the textbook homepage and fast-review links
to every textbook sidebar, refactor the root README, document general and
ML-course-specific skill usage, update only the shared files listed in Task 3,
run all specified automated checks, inspect the textbook and all three
canonical offline experiences on desktop and Android Chrome emulation, and
create the integration evidence document.

Commit only the Task 3 shared files with message
"feat: integrate Build Week learning companion showcase". Return:
1. integration commit SHA;
2. imported Regression and Classification commit SHAs;
3. exact verification results and skips;
4. textbook homepage/sidebar and companion visual-review findings;
5. expected Pages routes;
6. remaining human-led submission actions.

Do not draft or submit the Devpost entry. Submission wording, screenshots,
video, /feedback selection, and final rule review will be handled with the user
after integration.
```
