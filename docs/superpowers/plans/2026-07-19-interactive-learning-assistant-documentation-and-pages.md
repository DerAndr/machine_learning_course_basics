# Interactive Learning Assistant Documentation and Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the learning assistant in both course repositories, publish the EDA demo through the student GitHub Pages build, and merge the verified feature into both remote `main` branches.

**Architecture:** Keep `lecture_experiences/lecture_01_eda/index.html` as the only generated demo source. Extend the textbook preview builder to copy standalone experiences byte-for-byte into `site/_build/demos/lecture_01_eda/index.html`, then document the same live and offline paths across the repository. Apply shared changes to both repositories and add one teacher-only public-safety note after merging into the teacher integration branch.

**Tech Stack:** Python 3.12, pathlib/shutil, pytest, Ruff, Ty, Markdown, GitHub Actions, GitHub Pages, Git/GitHub CLI.

## Global Constraints

- The live student URL is exactly `https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/`.
- The offline source remains `lecture_experiences/lecture_01_eda/index.html`.
- The Pages builder copies standalone demo HTML byte-for-byte; it does not create or maintain a second generated demo.
- Content JSON is not copied into the Pages artifact.
- Do not modify `okf/`.
- Do not publish private solutions, grading artifacts, teacher quiz banks, or teacher notes.
- Preserve all teacher-only files and the teacher repository's private history.
- Do not force-push or rewrite history.
- A local failure blocks both `main` updates.
- A persistent live 404 blocks completion.

---

## File map

- `tools/build_textbook_preview.py`: copy committed standalone lecture experiences into the Pages artifact.
- `tests/test_textbook_preview.py`: prove the EDA demo is published byte-for-byte at the documented route.
- `tests/test_interactive_learning_assistant_docs.py`: keep the repository overview, guide, lecture page, and agent guide linked consistently.
- `README.md`: public feature discovery and live/offline links.
- `docs/interactive-lecture-learning-assistant.md`: complete learner and maintainer guide.
- `lectures/lecture_01_eda/README.md`: lecture-specific entry point to the demo.
- `AGENTS.md`: agent navigation, source-of-truth, generation, and validation guidance.
- `.github/workflows/build-textbook-preview.yml`: rebuild and deploy Pages when the skill, demo, or documentation changes.
- `.github/workflows/validate-okf.yml`: validate the shared learning-assistant tests on feature changes.
- `docs/publishing-model.md`: teacher-only public-safety and synchronization policy.

### Task 1: Publish standalone lecture experiences in the Pages artifact

**Files:**
- Modify: `tests/test_textbook_preview.py`
- Modify: `tools/build_textbook_preview.py`

**Interfaces:**
- Consumes: committed files matching `lecture_experiences/*/index.html`.
- Produces: `_copy_lecture_experiences(experiences: Path, build_dir: Path) -> None` and `site/_build/demos/lecture_01_eda/index.html`.

- [ ] **Step 1: Add a failing Pages-copy regression test**

Append this test to `tests/test_textbook_preview.py`:

```python
def test_textbook_preview_publishes_standalone_lecture_demo(tmp_path: Path) -> None:
    build_textbook_preview = load_builder()
    output = build_textbook_preview(output=tmp_path / "textbook")

    source = Path("lecture_experiences/lecture_01_eda/index.html")
    published = output / "demos" / "lecture_01_eda" / "index.html"

    assert published.is_file()
    assert published.read_bytes() == source.read_bytes()
    assert "Exploratory Data Analysis: Interactive Review" in published.read_text(
        encoding="utf-8"
    )
    assert not (output / "demos" / "content").exists()
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
& C:\projects\personal\ml-course\.venv\Scripts\python.exe -m pytest `
  tests/test_textbook_preview.py::test_textbook_preview_publishes_standalone_lecture_demo -q
```

Expected: failure because `demos/lecture_01_eda/index.html` does not exist.

- [ ] **Step 3: Add the deterministic copy helper**

Add this function before `build_textbook_preview` in
`tools/build_textbook_preview.py`:

```python
def _copy_lecture_experiences(experiences: Path, build_dir: Path) -> None:
    """Copy self-contained lecture demos into the Pages artifact."""
    if not experiences.is_dir():
        return
    for source in sorted(experiences.glob("*/index.html")):
        destination = build_dir / "demos" / source.parent.name / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
```

Extend the builder signature and call the helper after copying `site/assets`
and `site/data`:

```python
def build_textbook_preview(
    bundle: Path = Path("okf"),
    site: Path = Path("site"),
    output: Path | None = None,
    experiences: Path = Path("lecture_experiences"),
) -> Path:
    ...
    shutil.copytree(site / "assets", build_dir / "assets")
    shutil.copytree(site / "data", build_dir / "data")
    _copy_lecture_experiences(experiences, build_dir)
```

Add the CLI option and pass it by keyword:

```python
parser.add_argument(
    "--experiences",
    default=Path("lecture_experiences"),
    type=Path,
)
...
output = build_textbook_preview(
    bundle=args.bundle,
    site=args.site,
    output=args.output,
    experiences=args.experiences,
)
```

- [ ] **Step 4: Run focused tests and formatting**

Run:

```powershell
$python = 'C:\projects\personal\ml-course\.venv\Scripts\python.exe'
& $python -m pytest tests/test_textbook_preview.py -q
& $python -m ruff check tools/build_textbook_preview.py tests/test_textbook_preview.py
& $python -m ruff format --check tools/build_textbook_preview.py tests/test_textbook_preview.py
```

Expected: all commands exit zero and the published demo bytes match the source.

- [ ] **Step 5: Commit the Pages publication behavior**

```powershell
git add -- tools/build_textbook_preview.py tests/test_textbook_preview.py
git commit -m "feat: publish interactive lecture demos"
```

### Task 2: Document the learning assistant and EDA demo

**Files:**
- Create: `tests/test_interactive_learning_assistant_docs.py`
- Create: `docs/interactive-lecture-learning-assistant.md`
- Modify: `README.md`
- Modify: `lectures/lecture_01_eda/README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the learning-assistant skill, EDA payload, standalone HTML, and live Pages route.
- Produces: consistent public navigation for students, maintainers, and agents.

- [ ] **Step 1: Add a failing documentation-link contract**

Create `tests/test_interactive_learning_assistant_docs.py`:

```python
from pathlib import Path

LIVE_URL = (
    "https://derandr.github.io/machine_learning_course_basics/"
    "demos/lecture_01_eda/"
)
OFFLINE_PATH = "lecture_experiences/lecture_01_eda/index.html"
SKILL_PATH = ".agents/skills/ml-course-interactive-learning-assistant/SKILL.md"


def test_learning_assistant_documentation_is_discoverable() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/interactive-lecture-learning-assistant.md")
    lecture = Path("lectures/lecture_01_eda/README.md").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert guide.is_file()
    guide_text = guide.read_text(encoding="utf-8")
    assert LIVE_URL in readme
    assert LIVE_URL in guide_text
    assert OFFLINE_PATH in readme
    assert OFFLINE_PATH in guide_text
    assert OFFLINE_PATH in lecture
    assert SKILL_PATH in guide_text
    assert "lecture_experiences/content/" in agents
    assert "validate_lecture_site.py" in agents
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run:

```powershell
& C:\projects\personal\ml-course\.venv\Scripts\python.exe -m pytest `
  tests/test_interactive_learning_assistant_docs.py -q
```

Expected: failure because the guide and links do not exist.

- [ ] **Step 3: Create the complete guide**

Create `docs/interactive-lecture-learning-assistant.md` with these sections and
exact paths:

```markdown
# Interactive Lecture Learning Assistant

The learning assistant creates small, self-contained lecture review pages with
grounded explanations, interactive graphs, accessibility controls, funny
topic-related break prompts, and a 10-question quiz selected from one of three
difficulty banks.

## Try the EDA example

- [Open the live EDA review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/)
- Offline file: `lecture_experiences/lecture_01_eda/index.html`
- Editable payload: `lecture_experiences/content/lecture_01_eda.json`

Download or clone the repository and open the offline file directly. It uses no
server, CDN, external font, account, or network request.

## Learner controls

- Foundations, Applied, or Challenge question depth
- focus-friendly (ADHD-friendly) presentation
- color-blind-safe palette with non-color visual cues
- optional funny lecture-related break prompts
- interactive histogram, box plot, scatter plot, and missingness views
- immediate quiz feedback, answer review, progress, and retry
- keyboard navigation, visible focus, reduced motion, and static fallbacks

## Generate another lecture review

Use `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md`.
Ground content only in public course files, keep OKF read-only, and write a
payload following the skill's `references/content-contract.md`.

```powershell
$lectureSlug = 'lecture_02_data_preparation_part_1'
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_lecture_site.py `
  --content "lecture_experiences/content/$lectureSlug.json" `
  --template .agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html `
  --output "lecture_experiences/$lectureSlug/index.html"

uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py `
  "lecture_experiences/$lectureSlug/index.html"
```

## Publish through the textbook preview

```powershell
uv run python tools/build_textbook_preview.py
```

The builder copies each standalone lecture review such as
`lecture_experiences/lecture_01_eda/index.html` to the matching Pages route
`site/_build/demos/lecture_01_eda/index.html`. The standalone file remains the
source; generated files under `site/_build/` are not committed.

## Public-safety contract

Do not use private solutions, answer keys, grading data, teacher quiz banks,
teacher notes, or untracked workbooks. Every cited source must exist, and
lecture paths must belong to the selected lecture.
```

- [ ] **Step 4: Add public discovery links**

In `README.md`, add an `### Interactive lecture reviews` subsection after the
Interactive Textbook description. It must contain:

```markdown
### Interactive lecture reviews

Standalone lecture reviews complement the OKF textbook with focused,
offline-capable explanations, graphs, accessibility controls, and knowledge
checks.

- [Live EDA interactive review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/)
- [Offline EDA review](lecture_experiences/lecture_01_eda/index.html)
- [Learning-assistant guide](docs/interactive-lecture-learning-assistant.md)
```

In `lectures/lecture_01_eda/README.md`, add:

```markdown
## Interactive Review

- [Open the live EDA review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/)
- Offline file: [`lecture_experiences/lecture_01_eda/index.html`](../../lecture_experiences/lecture_01_eda/index.html)
```

- [ ] **Step 5: Extend agent navigation**

Update `AGENTS.md` in three places:

1. Under the interactive textbook layer, explain that standalone lecture
   reviews live under `lecture_experiences/` and are copied into
   `site/_build/demos/`.
2. Add a navigation sequence starting with
   `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md`, then the
   content JSON, generated HTML, guide, generator, and validator.
3. Add supporting-directory entries for
   `lecture_experiences/content/`, `lecture_experiences/lecture_01_eda/`, and
   the skill. Explain that other lecture directories follow the same structure.

Include this exact validation command:

```powershell
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py lecture_experiences/lecture_01_eda/index.html
```

- [ ] **Step 6: Run documentation tests and lint**

```powershell
$python = 'C:\projects\personal\ml-course\.venv\Scripts\python.exe'
& $python -m pytest `
  tests/test_interactive_learning_assistant_docs.py `
  tests/test_interactive_learning_assistant_skill.py -q
& $python -m ruff check tests/test_interactive_learning_assistant_docs.py
& $python -m ruff format --check tests/test_interactive_learning_assistant_docs.py
```

Expected: all commands exit zero.

- [ ] **Step 7: Commit shared documentation**

```powershell
git add -- README.md AGENTS.md `
  docs/interactive-lecture-learning-assistant.md `
  lectures/lecture_01_eda/README.md `
  tests/test_interactive_learning_assistant_docs.py
git commit -m "docs: add interactive lecture learning guide"
```

### Task 3: Make CI validate and deploy the feature

**Files:**
- Modify: `.github/workflows/build-textbook-preview.yml`
- Modify: `.github/workflows/validate-okf.yml`

**Interfaces:**
- Consumes: shared learning-assistant files and tests.
- Produces: Pages rebuilds and validation runs for every relevant feature change.

- [ ] **Step 1: Add workflow path triggers**

Add these entries to both push path lists and to the pull-request path list in
`validate-okf.yml`:

```yaml
      - "lecture_experiences/**"
      - "docs/interactive-lecture-learning-assistant.md"
      - "lectures/lecture_01_eda/README.md"
      - "README.md"
      - "AGENTS.md"
      - "tests/test_interactive_learning_assistant_docs.py"
      - "tests/test_interactive_learning_assistant_skill.py"
      - "tests/test_lecture_site_generator.py"
      - "tests/test_eda_lecture_experience.py"
```

- [ ] **Step 2: Add learning-assistant tests to workflow commands**

Append these paths to the Ruff and pytest commands in both workflows:

```text
tests/test_interactive_learning_assistant_docs.py
tests/test_interactive_learning_assistant_skill.py
tests/test_lecture_site_generator.py
tests/test_eda_lecture_experience.py
```

Keep `tools/build_textbook_preview.py` and `tests/test_textbook_preview.py` in
the existing commands.

- [ ] **Step 3: Validate workflow YAML and focused tests**

Run:

```powershell
$python = 'C:\projects\personal\ml-course\.venv\Scripts\python.exe'
& $python -c "import pathlib,yaml; [yaml.safe_load(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['.github/workflows/build-textbook-preview.yml','.github/workflows/validate-okf.yml']]"
& $python -m pytest `
  tests/test_textbook_preview.py `
  tests/test_interactive_learning_assistant_docs.py `
  tests/test_interactive_learning_assistant_skill.py `
  tests/test_lecture_site_generator.py `
  tests/test_eda_lecture_experience.py -q
```

Expected: YAML parses and all focused tests pass.

- [ ] **Step 4: Commit workflow integration**

```powershell
git add -- .github/workflows/build-textbook-preview.yml `
  .github/workflows/validate-okf.yml
git commit -m "ci: deploy interactive lecture demos"
```

### Task 4: Verify and publish the shared feature branch

**Files:**
- Verify only; no planned file changes.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: one verified shared commit tip available to both repositories.

- [ ] **Step 1: Run the full repository checks**

```powershell
$python = 'C:\projects\personal\ml-course\.venv\Scripts\python.exe'
& $python -m ruff format --check `
  src/mlcourse/okf_validation.py tools/validate_okf.py `
  tools/build_textbook_preview.py tests/test_okf_validation.py `
  tests/test_textbook_preview.py tests/test_smoke.py `
  tests/test_interactive_learning_assistant_docs.py `
  tests/test_interactive_learning_assistant_skill.py `
  tests/test_lecture_site_generator.py tests/test_eda_lecture_experience.py
& $python -m ruff check `
  src/mlcourse/okf_validation.py tools/validate_okf.py `
  tools/build_textbook_preview.py tests/test_okf_validation.py `
  tests/test_textbook_preview.py tests/test_smoke.py `
  tests/test_interactive_learning_assistant_docs.py `
  tests/test_interactive_learning_assistant_skill.py `
  tests/test_lecture_site_generator.py tests/test_eda_lecture_experience.py
& $python -m ty check src/mlcourse/okf_validation.py `
  tools/validate_okf.py tools/build_textbook_preview.py
& $python -m pytest -q
& $python tools/validate_okf.py okf/ --strict-warnings
& $python tools/build_textbook_preview.py
& $python .agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py `
  lecture_experiences/lecture_01_eda/index.html
```

Expected: every command exits zero.

- [ ] **Step 2: Verify local Pages output**

```powershell
$source = Get-FileHash lecture_experiences/lecture_01_eda/index.html -Algorithm SHA256
$published = Get-FileHash site/_build/demos/lecture_01_eda/index.html -Algorithm SHA256
if ($source.Hash -ne $published.Hash) { throw 'Published demo drifted from source' }
Select-String -Path site/_build/demos/lecture_01_eda/index.html `
  -Pattern 'Exploratory Data Analysis: Interactive Review'
git diff --name-only upstream/main..HEAD -- okf
```

Expected: hashes match, heading is found, and no `okf/` path is printed.

- [ ] **Step 3: Push the shared branch to both remotes**

```powershell
git push upstream codex/interactive-learning-assistant
git push origin codex/interactive-learning-assistant
```

Expected: both remote feature branches point to the same verified commit.

### Task 5: Fast-forward the student main branch and verify Pages

**Files:**
- No additional file changes.

**Interfaces:**
- Consumes: verified shared feature commit.
- Produces: updated `upstream/main` and live student demo.

- [ ] **Step 1: Confirm fast-forward safety**

```powershell
git fetch upstream main
git merge-base --is-ancestor upstream/main codex/interactive-learning-assistant
git status --porcelain
```

Expected: ancestor command exits zero and the worktree is clean.

- [ ] **Step 2: Push the verified feature tip to student main**

```powershell
git push upstream codex/interactive-learning-assistant:main
```

Expected: `upstream/main` advances without force.

- [ ] **Step 3: Verify the remote main commit**

```powershell
$expected = git rev-parse codex/interactive-learning-assistant
$actual = (git ls-remote upstream refs/heads/main).Split("`t")[0]
if ($expected -ne $actual) { throw 'Student main does not match verified feature' }
```

- [ ] **Step 4: Wait for student workflows**

```powershell
gh run list --repo DerAndr/machine_learning_course_basics `
  --branch main --limit 10
```

```powershell
$studentSha = git rev-parse codex/interactive-learning-assistant
$buildRun = gh run list --repo DerAndr/machine_learning_course_basics `
  --branch main --workflow 'Build Textbook Preview' --limit 1 `
  --json databaseId,headSha | ConvertFrom-Json
$validateRun = gh run list --repo DerAndr/machine_learning_course_basics `
  --branch main --workflow 'Validate OKF' --limit 1 `
  --json databaseId,headSha | ConvertFrom-Json
if ($buildRun.headSha -ne $studentSha -or $validateRun.headSha -ne $studentSha) {
  throw 'Student workflow runs do not match the pushed commit'
}
gh run watch $buildRun.databaseId `
  --repo DerAndr/machine_learning_course_basics --exit-status
gh run watch $validateRun.databaseId `
  --repo DerAndr/machine_learning_course_basics --exit-status
```

Expected: both runs finish successfully.

- [ ] **Step 5: Require the live demo to return HTTP 200**

```powershell
$url = 'https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/'
$response = Invoke-WebRequest -Uri $url -UseBasicParsing
if ($response.StatusCode -ne 200) { throw "Unexpected status $($response.StatusCode)" }
if ($response.Content -notmatch 'Exploratory Data Analysis: Interactive Review') {
  throw 'Live EDA heading missing'
}
```

Expected: HTTP 200 and the EDA heading is present.

### Task 6: Merge into teacher main with teacher-only documentation

**Files:**
- Modify only on teacher integration branch: `docs/publishing-model.md`

**Interfaces:**
- Consumes: verified shared feature commit and `origin/main`.
- Produces: a merge commit preserving teacher history plus a teacher-only policy commit.

- [ ] **Step 1: Create the teacher integration branch**

```powershell
$shared = git rev-parse codex/interactive-learning-assistant
git fetch origin main
git switch -c codex/teacher-interactive-learning-assistant origin/main
git merge --no-ff $shared -m "Merge interactive learning assistant"
```

Expected: the merge retains `docs/publishing-model.md` and all teacher-only
paths. Resolve conflicts by keeping public shared files from the feature and
teacher-only files from `origin/main`.

- [ ] **Step 2: Add the teacher-only publishing policy**

Append this section to `docs/publishing-model.md`:

```markdown
## Interactive Lecture Reviews

Interactive lecture payloads and generated HTML are public distribution
artifacts. Ground them only in public lecture files or read-only OKF sources.
Never include private solutions, grading artifacts, teacher quiz banks, answer
markers, teacher notes, or untracked assessment workbooks.

The student repository is the canonical public distribution point. Keep each
standalone lecture review synchronized with the copy published by the textbook
builder. For Lecture 01, the source is
`lecture_experiences/lecture_01_eda/index.html` and the Pages artifact is
`site/_build/demos/lecture_01_eda/index.html`.
```

- [ ] **Step 3: Commit the teacher-only documentation**

```powershell
git add -- docs/publishing-model.md
git commit -m "docs: define interactive demo publishing policy"
```

- [ ] **Step 4: Run the full teacher merged-state checks**

Run the complete command block from Task 4 Step 1, then verify:

```powershell
Test-Path docs/publishing-model.md
git diff --name-status origin/main..HEAD | Select-String '^D'
```

Expected: full checks pass, the publishing model exists, and no unexpected
teacher-only deletion is present.

- [ ] **Step 5: Incorporate a moved teacher main if necessary**

```powershell
git fetch origin main
git merge-base --is-ancestor origin/main HEAD
```

If the ancestor check fails, run:

```powershell
git merge --no-ff origin/main -m "Merge latest teacher main"
```

Then rerun Task 4 Step 1 before continuing.

- [ ] **Step 6: Push the teacher integration result to main**

```powershell
git push origin HEAD:main
```

Expected: `origin/main` advances without force.

- [ ] **Step 7: Verify teacher main and workflows**

```powershell
$expected = git rev-parse HEAD
$actual = (git ls-remote origin refs/heads/main).Split("`t")[0]
if ($expected -ne $actual) { throw 'Teacher main does not match integration tip' }
gh run list --repo DerAndr/machine_learning_course_teacher `
  --branch main --limit 10
```

Capture and watch the teacher workflow runs:

```powershell
$teacherSha = git rev-parse HEAD
$teacherRuns = gh run list --repo DerAndr/machine_learning_course_teacher `
  --branch main --commit $teacherSha --limit 20 `
  --json databaseId,name,headSha | ConvertFrom-Json
if (-not $teacherRuns -or ($teacherRuns.headSha | Select-Object -Unique) -ne $teacherSha) {
  throw 'Teacher workflow runs do not match the pushed commit'
}
foreach ($run in $teacherRuns) {
  gh run watch $run.databaseId `
    --repo DerAndr/machine_learning_course_teacher --exit-status
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

Expected: remote main matches and required checks succeed.

### Task 7: Final cross-repository audit

**Files:**
- Verify only.

**Interfaces:**
- Consumes: both updated main branches and deployed Pages.
- Produces: final evidence that documentation, code, and deployment agree.

- [ ] **Step 1: Compare shared feature paths**

```powershell
$paths = @(
  '.agents/skills/ml-course-interactive-learning-assistant',
  'lecture_experiences',
  'docs/interactive-lecture-learning-assistant.md',
  'tools/build_textbook_preview.py',
  'tests/test_textbook_preview.py'
)
foreach ($path in $paths) {
  git diff --exit-code upstream/main origin/main -- $path
}
```

Expected: shared feature paths are identical.

- [ ] **Step 2: Confirm repository-specific policy**

```powershell
git cat-file -e origin/main:docs/publishing-model.md
if (git cat-file -e upstream/main:docs/publishing-model.md 2>$null) {
  throw 'Teacher-only publishing model leaked into student main'
}
```

- [ ] **Step 3: Recheck the live route**

```powershell
$url = 'https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/'
$response = Invoke-WebRequest -Uri $url -UseBasicParsing
if ($response.StatusCode -ne 200) { throw 'Live demo is unavailable' }
if ($response.Content -notmatch 'Exploratory Data Analysis: Interactive Review') {
  throw 'Live demo content is incorrect'
}
```

- [ ] **Step 4: Record final branch and deployment evidence**

Report:

- the final student `main` SHA;
- the final teacher `main` SHA;
- successful workflow run URLs for both repositories;
- the live demo URL;
- full local test count;
- confirmation that shared paths match and teacher-only policy did not leak.
