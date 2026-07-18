# Interactive Lecture Learning Assistant Design

## Goal

Create a repository-local Codex skill that generates a small, self-contained interactive learning site for a selected machine-learning lecture. Each site helps students review core ideas, explore interactive graphs, and complete a 10-question knowledge check.

The first generated reference site covers Lecture 01, Exploratory Data Analysis (EDA).

## Scope

The implementation will add:

- a reusable `ml-course-interactive-learning-assistant` skill;
- a deterministic single-file site generator;
- a validator for generated sites;
- one complete EDA reference site;
- tests for the skill contract, generator, validator, and EDA experience.

The implementation will not delete, replace, or modify the OKF textbook. The skill may read OKF pages as supporting material, but lecture files remain the primary content source and `okf/` remains untouched.

## Output Contract

Each generated lecture experience is one `index.html` file containing:

- semantic HTML;
- inline CSS;
- inline JavaScript;
- embedded site configuration and learning content;
- inline SVG graphics and their accessible fallbacks;
- all quiz banks, answer explanations, and break prompts.

The page must open directly through a `file://` URL without a build step, local server, network request, external font, CDN, package install, or account.

The EDA reference output will be:

`lecture_experiences/lecture_01_eda/index.html`

Its reviewable authoring payload will be stored separately at:

`lecture_experiences/content/lecture_01_eda.json`

The JSON file is an authoring input, not a runtime dependency. The generated HTML remains independently portable.

## Skill Structure

The skill will live at:

`.agents/skills/ml-course-interactive-learning-assistant/`

Planned files:

- `SKILL.md`: triggering description and generation workflow;
- `agents/openai.yaml`: human-facing metadata;
- `assets/lecture-site-template.html`: reusable single-file template;
- `scripts/generate_lecture_site.py`: deterministic template renderer;
- `scripts/validate_lecture_site.py`: structural and offline validation;
- `references/content-contract.md`: required content, interaction, and accessibility schema.

The scripts will use the Python standard library unless an existing repository dependency provides a clear tested benefit.

## Generation Workflow

The skill will:

1. Identify the lecture from `lectures/index.yaml`.
2. Ask for generation defaults:
   - quiz depth: Foundations, Applied, or Challenge;
   - focus-friendly mode on or off;
   - color-blind palette on or off;
   - funny topic-related break prompts on or off.
3. Read the lecture README, lecture notes, `links.yaml`, example overview, and practical overview when available.
4. Read relevant public notebooks or OKF pages only when they improve grounding.
5. Write a structured JSON authoring payload containing explanations, graph specifications, practice checks, quiz banks, feedback, prompts, and source notes.
6. Pass that payload to the generator and produce one portable `index.html`.
7. Run the validator and browser smoke checks.

The skill must not use private solutions, answer keys, grading data, or untracked quiz workbooks as content sources.

## Learning Experience

### Setup

The opening panel lets a student choose:

- Foundations, Applied, or Challenge quiz depth;
- standard or focus-friendly presentation;
- default or color-blind-safe palette;
- break prompts on or off.

Generation-time choices establish defaults. Every choice remains switchable inside the finished page without regeneration.

Preferences and current progress may be stored in browser `localStorage`. No data leaves the browser, and the page must still work when storage is unavailable.

### Learn

The EDA reference site will explain a small set of foundational ideas grounded in Lecture 01. It will use short sections, definitions, interpretation guidance, and common mistakes rather than reproduce lecture prose.

Planned interactive views:

- histogram controls for understanding distribution shape and bin width;
- box plot controls for spread and outlier interpretation;
- scatter plot controls for association, scale, and misleading visual patterns;
- missing-data view for comparing counts and proportions.

Each view will ask the student to predict or interpret an outcome before or while changing a control. Each graph will include a text or table fallback.

### Quiz

The page embeds three separate quiz banks:

- 10 Foundations questions;
- 10 Applied questions;
- 10 Challenge questions.

Every mode always presents exactly 10 questions. Questions may use single-choice, multiple-choice, or interpretation formats supported by the same accessible interaction contract.

Each response produces immediate correctness feedback and a concise explanation. The final view shows score, missed concepts, answer review, and retry controls. Changing difficulty resets the active quiz after an explicit confirmation.

### Focus-Friendly Mode

The control will be labeled clearly as a focus-friendly, ADHD-friendly presentation option without making medical claims.

When enabled, it will:

- show one learning chunk or quiz question at a time;
- display progress;
- hide nonessential decorative and navigation elements;
- reduce motion;
- keep controls and instructions visually consistent;
- avoid countdown timers and time pressure.

### Break Prompts

Optional break prompts appear at deterministic learning milestones, such as after two concept sections and after question five. They will be short, funny, and related to the lecture topic.

Example tone: invite the student to “stretch those whiskers before the next box plot.” Prompts must not obscure content, interrupt keyboard focus, or require dismissal before continuing.

## Visual and Accessibility Design

The visual design will be minimal:

- restrained spacing and typography;
- clear hierarchy;
- high-contrast surfaces;
- limited decoration;
- responsive layout;
- no external fonts or icon libraries.

Color-blind mode will use a tested color-blind-safe palette based on established combinations such as Okabe–Ito. Meaning must never rely on color alone: marks also use labels, shapes, patterns, or line styles.

Required accessibility behavior:

- semantic landmarks and headings;
- keyboard-operable controls;
- visible focus indicators;
- properly associated labels and instructions;
- ARIA live regions only for concise dynamic status;
- reduced-motion support through both user settings and `prefers-reduced-motion`;
- readable graph summaries and table fallbacks;
- sufficient contrast in both palettes.

## Content Grounding

For each lecture, use this source order:

1. `lectures/<slug>/lecture_notes.md`;
2. `lectures/<slug>/README.md` and `links.yaml`;
3. public `lecture_examples/` material;
4. public practical README and student notebook;
5. relevant OKF concepts as read-only supporting sources.

Generated content must include embedded source notes naming the course files used. Unsupported claims, ambiguous answers, or missing evidence must be fixed before generation succeeds.

## Error Handling

Generation stops with a clear error when:

- the lecture slug is unknown;
- required lecture sources are missing;
- a quiz bank does not contain exactly 10 valid questions;
- a question lacks an answer or explanation;
- a graph lacks an accessible fallback;
- the template contains an external runtime dependency;
- required controls or accessibility hooks are absent.

At runtime, unavailable storage falls back to session-only state. JavaScript errors must leave explanations, graph fallbacks, and quiz content readable rather than presenting a blank page.

## Validation and Tests

Skill development will follow the required skill-writing RED, GREEN, REFACTOR workflow:

1. Run realistic generation scenarios without the new skill and record baseline gaps.
2. Write the smallest skill that corrects observed gaps.
3. Re-run scenarios with the skill and close discovered loopholes.

Automated checks will cover:

- valid skill metadata and discoverability;
- deterministic site generation;
- direct-file offline operation;
- absence of external runtime resources;
- all four runtime settings;
- exactly 10 questions per difficulty;
- answer and explanation completeness;
- graph controls and fallbacks;
- semantic structure, keyboard hooks, focus styling, ARIA status, and reduced motion;
- quiz scoring, review, reset, and retry;
- graceful behavior when browser storage is unavailable.

A browser smoke test will exercise the EDA page through setup, graph interaction, difficulty switching, quiz completion, scoring, and retry. Test output and validator results will provide concrete evidence for independent review, including review by Fable.

## Acceptance Criteria

Work is complete when:

- the new skill is discoverable and passes its scenario tests;
- the generator creates the EDA site deterministically;
- the EDA site opens directly offline as one HTML file;
- all learning, graph, quiz, focus, color-blind, and break-prompt behavior works;
- each difficulty contains exactly 10 grounded questions;
- accessibility and no-network checks pass;
- OKF content is unchanged;
- repository tests relevant to the change pass.
