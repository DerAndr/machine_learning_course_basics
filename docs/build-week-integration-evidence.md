# Build Week Learning Companion Integration Evidence

This record captures repository evidence for the three-learning-companion
showcase. It does not claim that the branch has been merged, deployed, or
submitted.

## 2026-07-20 learner-facing iteration

A visual review from the learner's point of view exposed three issues:

1. Regression and Classification repeated the same generic chart grammar, so
   distinct topics could feel like the same activity.
2. **Ignored class semantics.** Classification labels existed in the JSON, but
   the renderer assigned visual marks by point order.
3. The palette setting changed computed colors, but the standard and
   color-blind-safe blue/orange pairs were perceptually weak as two distinct
   modes.

The root cause was shared behavior. The **shared portable core** was fixed
instead of hand-editing three pages. It now validates semantic visualization
payloads, embeds pure calculations from
`visualization-models.js`, renders redundant non-color cues, and uses visibly
different purple/teal and blue/vermillion palettes. The ML-course adapter maps
each interaction to a lecture objective. EDA, Regression, and Classification
were then regenerated through the same deterministic pipeline.

The resulting companions now make their subject differences explicit:

- EDA retains its binning, IQR, association, and missingness explorations.
- Regression adds residual diagnostics, Ridge/Lasso regularization, and
  MAE/RMSE metric sensitivity.
- Classification adds threshold-dependent confusion outcomes and class-aware
  decision boundaries.

This is feedback made visible as a learner-facing iteration: review revealed a
repeated interaction pattern, the portable abstraction was improved, and every
downstream artifact was regenerated.

### Evidence status for this iteration

| Check | Current evidence |
|---|---|
| Deterministic regeneration | Deterministic regeneration: completed on 2026-07-20. All three payloads regenerated into temporary files with byte-identical SHA-256 hashes to their committed artifacts. |
| Automated | Local full suite: completed (222 passed, 1 skipped). Ruff format/check, the three Node behavior suites (13 tests), strict OKF validation, and the preview build are recorded below. |
| Validators and preview build | Validators and preview build: completed on 2026-07-20 for all three companions, strict OKF validation, and the textbook preview. |
| Browser | Browser acceptance: completed on 2026-07-20 in storage-disabled, sandboxed frames for all three companions. Scripts were allowed, but origin and `localStorage` were unavailable; each frame rendered its heading and default UI, and the browser console recorded zero warnings or errors. Separate browser checks completed control-state retention, both palette modes, and no horizontal overflow at 390px. |
| GitHub Actions | Succeeded on public `main` at `a9fd6d9ec61f5c05981734846d858572413f5f92`: [Validate OKF run 29729743749](https://github.com/DerAndr/machine_learning_course_basics/actions/runs/29729743749) and [Build Textbook Preview run 29729739505](https://github.com/DerAndr/machine_learning_course_basics/actions/runs/29729739505). |
| GitHub Pages | Succeeded from Build Textbook Preview run `29729739505` at the published SHA. The root and all three companion routes returned HTTP 200 with the markers recorded below. |

## Integrated commits

- `68c3b2b` — cherry-pick of `dd03784`, Regression companion
- `642657c` — cherry-pick of `5c2a24d`, Classification Part 1 companion
- Common integration base: `c276b9d` on `basics`
- Integration branch: `codex/build-week-showcase`

Each imported commit owns exactly one grounded JSON payload, one generated
standalone HTML artifact, and one topic-specific deterministic test.

The semantic revision and subsequent local hardening include pure visualization
models, semantic schemas and renderers, Classification and Regression payloads,
authoring guidance, regenerated EDA, CI coverage, validation refinements, and
public evidence updates. The complete local history is available through
`git log --oneline`.

## Artifact inventory

| Slug | Title | Quiz inventory | SHA-256 |
|---|---|---:|---|
| `lecture_01_eda` | Exploratory Data Analysis: Interactive Review | 10 Foundations + 10 Applied + 10 Challenge | `B4B5E063EA25EDE63AE224CC2CBA4C9DA4B05ACF619ADEAA55F82E41FA385EE7` |
| `lecture_04_regression` | Regression: Interactive Review | 10 Foundations + 10 Applied + 10 Challenge | `127EE8C1E4944D9ADA23D815FBC8CF0900A67CC4A46A863BBE96B15F12EE2A0E` |
| `lecture_05_classification_part_1` | Classification Part 1: Interactive Review | 10 Foundations + 10 Applied + 10 Challenge | `35019B11E13CE7D0CD8A9C032942E6CA9AF8DC0FD8496EF777B87C95BE84E6F4` |

The canonical artifacts are the committed
`lecture_experiences/<slug>/index.html` files. Preview publication copies them
byte-for-byte. These hashes describe the current local semantic revision and
have been verified by the completed deterministic regeneration check.

## Source-policy verification

- Regression sources are restricted to
  `lectures/lecture_04_regression/...` and read-only `okf/...` paths.
- Classification sources are restricted to
  `lectures/lecture_05_classification_part_1/...` and read-only `okf/...`
  paths.
- Topic tests verify all cited paths, 30 unique questions, grounded concept
  associations, valid answers, explanatory feedback, meaningful graph
  controls, static fallbacks, and unchanged OKF content during generation.
- `git diff --exit-code basics -- okf` was empty before the shared textbook
  homepage copy was added.
- Both imported artifacts passed the offline validator before shared edits.

## Historical integration test evidence

Run on 2026-07-19 from the integration worktree:

- Ruff format check: passed.
- Ruff lint: passed.
- `ty` type check: passed.
- Full pytest suite: **161 passed, 1 optional browser test skipped**.
- Strict OKF validation: **7 concepts and 7 indexes; 0 errors, 0 warnings**.
- Textbook preview build: passed.
- EDA, Regression, and Classification offline validators: all reported
  `VALID`.
- `git diff --check`: passed.

The integration tests cover homepage cards, the all-page fast-review sidebar,
correct nested relative links, matching payload/artifact discovery,
path-specific malformed-payload errors, byte-identical demo copies, guide
prompts, README structure, and lecture-level links.

## Local semantic-revision evidence

Run on 2026-07-20 for the final local publication pass:

- The portable visualization model suite passed **10 Node tests**.
- Focused Classification generator and payload checks passed **132 tests**;
  its generated artifact reported `VALID`.
- Focused Regression generator and payload checks passed **132 tests**; its
  generated artifact reported `VALID`.
- Core and adapter contract checks passed **34 tests with 1 optional skip**;
  both skills passed their quick validators.
- The combined quiz, visualization-model, and visualization-control-state
  suite passed **13 Node tests**.
- The local full Python suite passed **222 tests with 1 optional skip**.
- Both CI workflow files parsed as YAML; Ruff format and lint, strict OKF
  validation, the textbook preview build, and `git diff --check` passed.
- Browser acceptance completed in storage-disabled, sandboxed frames for EDA,
  Regression, and Classification. Scripts were allowed, but origin and
  `localStorage` were unavailable; each rendered its heading and default UI,
  and the browser console recorded zero warnings or errors. Separate browser
  checks completed control-state retention, both palette modes, and no
  horizontal overflow at 390px.

These completed local checks were followed by successful public GitHub Actions
and GitHub Pages deployment at
`a9fd6d9ec61f5c05981734846d858572413f5f92`.

## Deterministic regeneration status

Each semantic payload has been generated through the course wrapper, and the
topic tests assert deterministic output against its committed artifact. On
2026-07-20, all three were regenerated into temporary verification files and
each SHA-256 hash matched its committed artifact. The artifact inventory
therefore records the verified current local bytes; it is not a deployment
claim.

## Historical desktop visual review

- The textbook homepage shows three fast-review cards and a separate
  three-link sidebar group.
- A deep Classification Metrics textbook page shows the same group with
  working `../../demos/<slug>/index.html` routes and no horizontal overflow.
- The Classification companion uses the available desktop width without
  horizontal overflow; learning settings, progress, and content cards remain
  readable.
- The browser console recorded no errors while reviewing the textbook and
  companion routes.

## Historical mobile visual review

- The textbook homepage and deep page collapse to one column with a static
  sidebar at the mobile breakpoint.
- The homepage retains all three cards and all three sidebar links.
- EDA and Classification fit a 375-pixel content viewport without horizontal
  overflow.
- Headings, learning settings, progress, concepts, graph fallbacks, and quiz
  controls remain readable and operable.

## Textbook discovery review

- `LearningExperience` descriptors are loaded from sorted matching
  `lecture_experiences/<slug>/index.html` and
  `lecture_experiences/content/<slug>.json` pairs.
- Missing payloads, invalid JSON, malformed `meta`, empty experience IDs, and
  empty titles stop the build with path-specific errors.
- The copy step remains byte-for-byte and does not publish the `content/`
  directory.
- Generated cards appear only on the textbook homepage; the fast-review
  navigation group appears on every rendered textbook page.
- Focus-visible outlines are defined for sidebar links and review cards.

## README and authoring-guide review

- The repository README now starts with four clear learning routes, followed
  by fast reviews, the complete 19-entry course map, skill selection, concise
  setup, contribution links, repository map, and licensing.
- The operational guide separates the portable core from the ML-course
  adapter, provides exact copy-ready prompts for both, and documents the full
  sources-to-publication pipeline.
- Root, operational, and matching lecture READMEs include live and offline
  routes for all three companions.

## Verified public deployment

On 2026-07-20, the public repository `main` at
`a9fd6d9ec61f5c05981734846d858572413f5f92` completed
[Validate OKF run 29729743749](https://github.com/DerAndr/machine_learning_course_basics/actions/runs/29729743749)
and [Build Textbook Preview run 29729739505](https://github.com/DerAndr/machine_learning_course_basics/actions/runs/29729739505).
The following live checks all returned HTTP 200:

| Route | Confirmed marker |
|---|---|
| `https://derandr.github.io/machine_learning_course_basics/` | `Fast interactive reviews` |
| `https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/` | `Palette` |
| `https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/` | `Ridge` |
| `https://derandr.github.io/machine_learning_course_basics/demos/lecture_05_classification_part_1/` | `Decision threshold` |

## Remaining human-led submission work

- Select primary /feedback session ID.
- Capture final screenshots.
- Record/upload public <3min video.
- Write Devpost description in submitter own voice.
- Complete official rule/announcement checklist.
- Submit before deadline.
