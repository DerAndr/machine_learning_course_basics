# Build Week Learning Companion Integration Evidence

This record captures repository evidence for the three-learning-companion
showcase. It does not claim that the branch has been merged, deployed, or
submitted.

## Integrated commits

- `68c3b2b` — cherry-pick of `dd03784`, Regression companion
- `642657c` — cherry-pick of `5c2a24d`, Classification Part 1 companion
- Common integration base: `c276b9d` on `basics`
- Integration branch: `codex/build-week-showcase`

Each imported commit owns exactly one grounded JSON payload, one generated
standalone HTML artifact, and one topic-specific deterministic test.

## Artifact inventory

| Slug | Title | Quiz inventory | SHA-256 |
|---|---|---:|---|
| `lecture_01_eda` | Exploratory Data Analysis: Interactive Review | 10 Foundations + 10 Applied + 10 Challenge | `1DE157D38E43E105BA3222D6D7370DD898A04E3F7F3C37ED80062B3F27CFCF1E` |
| `lecture_04_regression` | Regression: Interactive Review | 10 Foundations + 10 Applied + 10 Challenge | `FD14552AA19530DFC1F88DF04F3EEAD1B8CA352F561454DE1C0951EA9BCF8631` |
| `lecture_05_classification_part_1` | Classification Part 1: Interactive Review | 10 Foundations + 10 Applied + 10 Challenge | `7BC8B13BE5F860E8B167EBBD963C3981E70BB0F400F18B9C7BB61686FB91349E` |

The canonical artifacts are the committed
`lecture_experiences/<slug>/index.html` files. Preview publication copies them
byte-for-byte.

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

## Automated test evidence

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

## Deterministic regeneration

All three payloads were regenerated with the course wrapper and portable core
template into a separate verification directory. Each generated file matched
the committed artifact byte-for-byte, producing the SHA-256 values in the
artifact inventory.

## Desktop visual review

- The textbook homepage shows three fast-review cards and a separate
  three-link sidebar group.
- A deep Classification Metrics textbook page shows the same group with
  working `../../demos/<slug>/index.html` routes and no horizontal overflow.
- The Classification companion uses the available desktop width without
  horizontal overflow; learning settings, progress, and content cards remain
  readable.
- The browser console recorded no errors while reviewing the textbook and
  companion routes.

## Mobile visual review

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

## Published route expectations

After this branch is merged to `main` and the existing Pages workflow succeeds,
the expected routes are:

- `https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/`
- `https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/`
- `https://derandr.github.io/machine_learning_course_basics/demos/lecture_05_classification_part_1/`

These are route expectations, not evidence of the post-merge deployment.

## Remaining human-led submission work

- Verify deployed Pages after merge to main.
- Select primary /feedback session ID.
- Capture final screenshots.
- Record/upload public <3min video.
- Write Devpost description in submitter own voice.
- Complete official rule/announcement checklist.
- Submit before deadline.
