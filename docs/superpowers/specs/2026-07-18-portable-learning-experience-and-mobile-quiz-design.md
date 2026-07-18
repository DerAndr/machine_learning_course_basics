# Portable Learning Experience Builder and Mobile Quiz Fix

## Goal

Turn the existing ML-course learning assistant into a layered system that can
work in any repository or knowledge base while preserving all ML-course and OKF
rules. Fix the reported Chrome-on-Android quiz and progress defects in the
shared runtime.

## Architecture

### Portable core skill

Add `interactive-learning-experience-builder`, a domain-neutral skill that:

1. reads repository instructions before acting;
2. inventories available knowledge sources, build tools, tests, and publishing
   conventions;
3. records its findings in a repository-specific experience specification;
4. grounds explanations, graphs, quiz questions, and answer feedback in named
   sources;
5. generates and validates a fully self-contained interactive HTML experience;
6. optionally creates a thin repository adapter skill for recurring workflows.

The core must not assume ML, lectures, OKF, `uv`, Python, GitHub Pages, or a
particular directory layout. It will own the reusable template, generator,
validator, accessibility rules, and interaction contract.

### Repository adapter

Keep `ml-course-interactive-learning-assistant` as a thin adapter. It will add:

- lecture lookup through `lectures/index.yaml`;
- the existing ML-course source hierarchy;
- the rule that `okf/` is read-only supporting context;
- public-safety restrictions;
- course-specific output, validation, and deployment conventions.

The adapter will point to the portable core workflow and must not duplicate the
core implementation.

### Textbook contributor skill

Keep `ml-course-textbook-contributor` course-specific. Add mobile interaction
quality requirements for learning experiences and browser labs.

### Global installation

Install the same portable core skill under the user's global Codex skills
directory so it is discoverable from unrelated repositories. Keep the
repository copy as the version-controlled canonical source.

## Context adaptation

Before generation, the portable skill creates a short experience specification
covering:

- learner and learning goal;
- authoritative and supporting sources;
- repository constraints and excluded/private sources;
- requested depth and accessibility settings;
- output location and self-contained/offline requirements;
- available validation, browser-testing, and publishing commands.

Repository instructions, build tooling, and publishing commands are optional.
For a standalone knowledge bundle with no repository, the profile records the
knowledge root and stable source identifiers instead. Source identifiers may be
repository-relative paths, stable URLs, or explicit knowledge-base identifiers;
the generated page never fetches them at runtime.

For a one-off topic, this specification is sufficient. For a recurring
repository workflow, the skill may create a thin adapter skill containing only
stable repository-specific rules. It must not create a separate skill for every
lecture or individual page.

## Quiz state machine

Each question tracks:

- number of submitted attempts;
- whether the first submitted attempt was correct;
- whether the question is complete.

Transitions:

1. Selecting or changing an answer clears stale validation feedback.
2. Submitting no answer shows a prompt and remains on the question.
3. Submitting a wrong answer increments attempts, shows explanatory feedback,
   and remains on the same question. The Check button stays available.
4. Submitting a correct answer increments attempts, marks the question
   complete, disables its inputs, and reveals Next question.
5. Only Next question can advance after completion.

Progress counts completed questions, not attempts. Results report first-attempt
correctness and total attempts, preserving diagnostic value while allowing
students to learn through retries.

Whole-quiz Retry preserves learner settings but resets the question index,
attempt counts, first-attempt flags, completion flags, feedback, results,
progress, disabled inputs, and Check/Next visibility.

## Sticky progress

The progress panel uses `position: sticky`, a safe-area-aware top offset, and a
high enough stacking layer to remain visible without covering controls. On
small screens it becomes compact. Focused content receives appropriate scroll
offset so the sticky panel does not obscure it.

## Verification

Automated tests must cover:

- wrong answers do not complete or advance a question;
- changing an answer clears stale feedback;
- correct answers are the only quiz submissions that expose Next question;
- progress counts completed questions;
- results retain first-attempt accuracy and total attempts;
- the progress panel has the sticky/mobile contract;
- the portable skill contains no ML-course path assumptions;
- the adapter preserves lecture and OKF rules.

The existing adapter regression suite must remain passing. Adapter parity tests
must explicitly retain three banks of exactly ten questions, the four learner
defaults, deterministic single-file `file://` output, complete static
no-JavaScript explanations and quiz review, chart fallbacks, keyboard and
reduced-motion behavior, storage fallback, course source restrictions, answer
review, and whole-quiz Retry.

A portability smoke test must copy the canonical core into an unrelated
temporary repository with no ML-course layout or build tooling, generate a
valid offline HTML file from non-ML knowledge, and validate it.

Browser verification must use Android Chrome or Chrome device emulation with a
mobile viewport, touch input, Android user agent, device pixel ratio, and
dynamic viewport behavior when the available runner supports those controls.
It must cover `wrong → different wrong → correct → next`, whole-quiz Retry,
scrolling with the sticky progress panel, all three question depths,
focus-friendly mode, color-blind mode, and disabled storage. Feedback,
Check/Next controls, and newly focused questions must remain visible and
unobscured by the sticky panel.

The repository copy is canonical. Global installation is a deterministic sync
from that directory. Verification compares the installed `SKILL.md`, template,
generator, validator, and references with the canonical files so drift is
reported rather than silently accepted.
