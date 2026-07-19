# Build Week Learning Companion Showcase Design

## Goal

Prepare a convincing OpenAI Build Week Education-track showcase for the
general-purpose `interactive-learning-experience-builder` skill.

The showcase will demonstrate that one portable Codex workflow can turn named,
trusted repository sources into grounded, accessible, self-contained learning
companions. The existing ML course is the demonstration repository, not the
limit of the product.

## Submission Positioning

Present the project as a reusable learning-companion system with four distinct
layers:

1. a domain-neutral Codex skill that discovers repository context and enforces
   a portable content contract;
2. an optional thin repository adapter that applies stable local source,
   safety, output, and publishing rules;
3. grounded, reviewable JSON payloads for individual learning experiences; and
4. deterministic, offline HTML companions with explanations, interactive
   visualizations, accessibility controls, corrective quizzes, and static
   fallbacks.

The ML-course adapter and three lecture companions prove that the same core can
be reused across topics without copying or changing the runtime.

The submission should emphasize the transformation:

```text
trusted repository sources
  -> context discovery and source policy
  -> portable skill plus repository adapter
  -> grounded JSON payload
  -> deterministic generator and validator
  -> accessible offline learning companion
```

## Scope

Keep the existing EDA companion as the reference implementation and add exactly
two companions:

- Lecture 04: Regression
- Lecture 05: Classification Part 1

This produces three visibly different demonstrations while leaving enough time
for integration, visual review, documentation, video preparation, and Devpost
submission checks.

The integration also makes the companions discoverable from the repository
README and the rendered textbook, and explains how maintainers use the portable
skill and ML-course adapter to create further materials.

Do not add another lecture, a new visualization type, a new runtime dependency,
or a new repository adapter for this submission.

## Existing Foundation

The implementation may rely on the current repository architecture:

- `.agents/skills/interactive-learning-experience-builder/` is the canonical
  portable core.
- `.agents/skills/ml-course-interactive-learning-assistant/` is the thin
  ML-course adapter.
- `lecture_experiences/content/lecture_01_eda.json` is the reference payload.
- `lecture_experiences/lecture_01_eda/index.html` is the canonical offline
  reference artifact.
- `tools/build_textbook_preview.py` discovers and copies committed
  `lecture_experiences/*/index.html` files into matching Pages demo routes.
- The current preview does not link those routes from its homepage or sidebar,
  so copied companions are effectively hidden unless a learner already knows
  their URLs.
- Existing tests cover the core, adapter, portability, EDA payload, generated
  artifact, quiz behavior, and publication convention.

At design time, the focused verification suite reported 33 passing tests and
one skipped test. The committed EDA HTML passed the offline validator. The
repository's documented `uv` executable was not available on the shell PATH,
but `.venv/Scripts/python.exe` provided the working test environment.

## Parallel Work Design

Run the two lecture tasks concurrently only in isolated Git worktrees or
branches. Each task has exclusive file ownership and must not edit shared
runtime, documentation, build, or configuration files.

### Regression task ownership

The regression agent owns only:

- `lecture_experiences/content/lecture_04_regression.json`
- `lecture_experiences/lecture_04_regression/index.html`
- `tests/test_regression_lecture_experience.py`

It must ground the experience only in public, permitted Lecture 04 sources and
read-only supporting OKF sources allowed by the course adapter. It must use the
existing portable content contract, visualization types, template, generator,
validator, learner settings, and quiz behavior.

### Classification task ownership

The classification agent owns only:

- `lecture_experiences/content/lecture_05_classification_part_1.json`
- `lecture_experiences/lecture_05_classification_part_1/index.html`
- `tests/test_classification_part_1_lecture_experience.py`

It must ground the experience only in public, permitted Lecture 05 sources and
read-only supporting OKF sources allowed by the course adapter. It must use the
existing portable content contract, visualization types, template, generator,
validator, learner settings, and quiz behavior.

### Parallel-task constraints

Both agents must:

- read `AGENTS.md`, the portable core skill, the ML-course adapter, context
  discovery reference, content contract, and EDA reference payload before
  authoring;
- preserve the existing portable core and adapter without modification;
- preserve `okf/` byte-for-byte;
- avoid private solutions, teacher materials, answer keys, grading data,
  unpublished drafts, and untracked quiz workbooks;
- include three quiz banks named `foundations`, `applied`, and `challenge`,
  with exactly ten grounded questions in each bank;
- use stable unique identifiers and connect every question to a declared
  concept;
- use only the existing `histogram`, `boxplot`, `scatter`, and `missingness`
  visualization schemas;
- include readable visualization fallbacks and complete static no-JavaScript
  content;
- generate the HTML through the course wrapper rather than editing generated
  HTML by hand;
- prove deterministic regeneration against the committed artifact;
- validate the resulting HTML and run the relevant focused tests; and
- avoid staging, deleting, or modifying unrelated existing work, including
  `catboost_info/` and `quizzes/`.

## Experience Design

### Regression companion

The regression experience should teach a coherent subset of Lecture 04 rather
than compressing the entire lecture. Its concept arc should cover:

1. regression targets and model families;
2. fitted values, residuals, and ordinary least squares;
3. assumptions as diagnostic questions rather than automatic guarantees;
4. multicollinearity and coefficient instability;
5. Ridge, Lasso, and the role of feature scaling;
6. MAE, MSE, RMSE, and R-squared trade-offs;
7. residual analysis;
8. overfitting, validation, and interpretation limits.

Its visualizations should reuse the existing schemas for an illustrative
fitted relationship, residual distribution, influential or unusual residuals,
and missing-data context where it materially supports the lesson.

### Classification companion

The classification experience should teach a coherent subset of Lecture 05
rather than cataloging every model. Its concept arc should cover:

1. binary, multiclass, multilabel, and ordinal targets;
2. KNN intuition, distance, scaling, and the role of `k`;
3. tree splits, impurity, depth, and overfitting;
4. logistic scores, probabilities, and decision thresholds;
5. confusion-matrix outcomes;
6. precision, recall, F-scores, and class imbalance;
7. ROC/AUC and probability-sensitive evaluation;
8. model choice and interpretability trade-offs.

Its visualizations should reuse the existing schemas for illustrative distance
or score distributions, class-separated points, error distributions, or
missing-data context. A new threshold-specific visualization is explicitly out
of scope; threshold learning can be grounded through explanation, quiz
scenarios, and an appropriate existing visualization.

## Integration Phase

After both parallel branches are complete, run one sequential integration task.
The integration task may edit shared files and must:

1. review each branch for source grounding, payload correctness, generated-file
   determinism, visual clarity, and compliance with its file-ownership
   boundary;
2. integrate both lecture contributions and resolve only genuine integration
   issues;
3. refactor the repository README into a cleaner course-first entry point with
   clear learning routes and prominent fast-review links;
4. explain general and ML-course-specific companion authoring in the
   operational guide;
5. make all matched companions discoverable through generated cards on the
   textbook homepage and a dedicated fast-review group in every textbook
   sidebar;
6. verify that the preview build copies all three committed companions to their
   expected demo routes;
7. run the focused learning-companion tests, full repository tests, strict
   source validation, offline validators, and preview build;
8. inspect all three experiences on desktop and mobile, including settings,
   visualizations, fallbacks, quiz retry, keyboard focus, reduced motion, and
   storage-disabled behavior; and
9. leave a concise evidence record for submission preparation.

The integration task must not redesign the portable core merely to make the two
new payloads easier to author.

## Textbook Discovery Design

The textbook preview must discover companions from existing canonical inputs
rather than hardcoding three lecture names in the renderer.

At build time:

1. enumerate `lecture_experiences/*/index.html`;
2. use the directory name as the stable lecture slug;
3. require a matching
   `lecture_experiences/content/<lecture_slug>.json`;
4. read `meta.experience_id` and `meta.title` from that payload;
5. construct the matching output route
   `demos/<lecture_slug>/index.html`; and
6. sort companion descriptors deterministically by slug.

Only complete JSON/HTML pairs appear in discovery navigation. If a committed
HTML artifact has no matching payload, or a matched payload is malformed or
lacks required metadata, the preview build must fail with a clear path-specific
message rather than silently hiding the companion.

The rendered textbook homepage must contain a **Fast interactive reviews**
section after the introductory OKF content. It explains that companions are
short, focused, offline-capable practice experiences that complement the deeper
textbook. Each generated card shows the payload title, identifies the artifact
as a fast interactive review, and links to its copied demo route.

Every rendered textbook page must include a distinct **Fast reviews** group in
the sidebar below normal OKF textbook navigation. Links must be calculated
relative to the current output page so they work from both the root homepage and
deeply nested concept pages.

Reuse the existing visual language for relationship cards where possible.
Add only the CSS selectors needed to separate the fast-review navigation group,
preserve visible focus and hover states, and keep the combined sidebar usable on
mobile. Do not add client-side fetching or another JavaScript runtime.

Update `okf/index.md` with durable explanatory text about the distinction
between the interactive textbook and fast interactive reviews. The generated
links remain renderer-owned because their routes derive from committed
companion artifacts; `okf/index.md` must remain useful when read directly on
GitHub.

## Repository README Design

Keep the ML course, not the hackathon, as the public repository's primary
identity. Refactor the README around these sections:

1. **What this repository offers.** One short explanation of the course and its
   student-facing source materials.
2. **Choose how to learn.** Four explicit entry points:
   interactive textbook, fast interactive reviews, full lecture materials, and
   hands-on notebooks.
3. **Fast interactive reviews.** Prominent live and offline links for EDA,
   Regression, and Classification Part 1, plus a one-paragraph explanation of
   their focused explanation-exploration-quiz-feedback loop.
4. **Course map.** Preserve the complete lecture table.
5. **Create interactive learning materials.** Explain when to use the portable
   skill versus the ML-course adapter and link to the operational guide.
6. **Local setup, contributing, repository map, and licensing.** Retain
   essential commands and authoritative links while removing duplicated or
   malformed prose already covered by detailed guides.

Do not turn the README into a Build Week landing page. Submission-specific
claims, deadlines, and judging language remain outside the durable student
README.

## Skill Usage and Authoring Guidance

Use `docs/interactive-lecture-learning-assistant.md` as the long-form
operational guide. Extend it with:

- a **Choose the right skill** comparison:
  - `interactive-learning-experience-builder` for a grounded one-off experience
    in any repository or knowledge base;
  - `ml-course-interactive-learning-assistant` together with the portable core
    when creating a review for a lecture in this repository;
- the existing exact default prompts from each skill's `agents/openai.yaml`;
- one concrete general-purpose invocation example;
- one concrete ML-course invocation example using an existing lecture slug;
- the complete authoring flow:
  sources, context profile, experience specification, grounded JSON payload,
  generation, offline validation, browser review, and optional publication;
- the ownership rule that the core contains reusable workflow/runtime behavior,
  the adapter contains stable repository policy, and individual experiences
  contain topic-specific payloads and generated artifacts; and
- a short verification checklist covering source policy, deterministic
  regeneration, static fallbacks, quiz behavior, accessibility, and publishing.

Keep the root README summary concise and link to this guide rather than
duplicating its commands and policy details.

## Submission Preparation Phase

Submission preparation begins only after integration and visual verification.
It is a separate review-and-packaging activity, not a third feature-building
stream.

Prepare:

- a concise problem statement describing the difficulty of turning trusted
  educational sources into interactive practice without losing provenance,
  accessibility, portability, or human reviewability;
- a product explanation centered on the portable skill and optional adapter,
  with the ML course as proof;
- repository setup, installation, generation, validation, and testing
  instructions that a judge can follow;
- public demo links or direct offline artifacts for all three companions;
- screenshots showing the source-to-companion workflow and the learner
  experience;
- a demo video of less than three minutes with audible explanation of what was
  built and how Codex and GPT-5.6 were used;
- a short architecture view and a clear statement of the Build Week work;
- the required `/feedback` session ID from the primary build task;
- Devpost category, repository URL, testing instructions, and any plugin or
  developer-tool installation details; and
- a final compliance checklist against current official rules and
  announcements.

The video should demonstrate one complete transformation in detail, then show
the other two companions briefly as evidence of repeatability. It should not
attempt to teach all three lectures or tour every feature.

## Verification and Acceptance

The showcase is ready for submission when:

- the EDA, regression, and classification payloads each pass the course source
  policy;
- all three generated artifacts pass the offline validator;
- regeneration is byte-for-byte deterministic for every committed artifact;
- every experience contains three banks of exactly ten grounded questions;
- the parallel lecture tasks did not modify `okf/`, and integration changed
  only `okf/index.md` to explain the discovery surface;
- no task used private material or modified unrelated user work;
- focused and full test suites pass in the integrated branch;
- the preview build publishes all three routes from the canonical committed
  HTML files;
- the rendered textbook homepage contains cards for all three companions;
- every rendered textbook sidebar contains working relative fast-review links;
- the repository README clearly distinguishes the textbook, fast reviews,
  lecture materials, and notebooks;
- the operational guide explains both skills and the complete authoring flow;
- desktop and mobile visual review finds no blocking accessibility or
  interaction defect;
- judge-facing setup and testing instructions work from a clean checkout;
- the demo video is viewable without authentication, shorter than three
  minutes, and contains the required Codex and GPT-5.6 explanation; and
- the Devpost draft contains every required field before the submission
  deadline.
