# Interactive Lecture Learning Assistant

This is the operational guide for creating, validating, and publishing
learning companions. For the conceptual layers, ownership boundaries, and
portability model, read
[Learning Companions Architecture](learning-companions-architecture.md).
Students who want copy-ready prompts and local Codex setup should start with
the
[student learning-companion quickstart](student-learning-companion-quickstart.md).

A learning companion is a small, self-contained review with grounded
explanations, meaningful visual controls, accessibility settings, optional
break prompts, and a 10-question quiz selected from one of three difficulty
banks.

## Choose the right skill

| Context | Use |
|---|---|
| Any repository or knowledge base | `.agents/skills/interactive-learning-experience-builder/SKILL.md` |
| This ML course | `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md` with the portable core |

The portable core owns the domain-neutral experience specification, grounded
JSON content contract, deterministic single-file generator, quiz behavior, and
offline validator. The ML-course adapter owns only stable course rules: the
public source hierarchy, exclusions, lecture-scoped paths, output locations,
and Pages route convention. A lecture payload owns the topic-specific content
and source citations.

## General-purpose workflow

Use this prompt:

> Use $interactive-learning-experience-builder to create a grounded, offline interactive learning experience from this repository's knowledge sources.

Before authoring, record a context profile with the learner and goal, named
authoritative and supporting sources, excluded or private material,
accessibility defaults, output path, and available validation or publishing
commands. A repository does not need an `AGENTS.md`, build tool, or publishing
configuration: stable source identifiers and an output location are enough.
Add a thin repository adapter only when the same local constraints recur.

## ML-course workflow

Use this prompt:

> Use $ml-course-interactive-learning-assistant with $interactive-learning-experience-builder to create a grounded, accessible, self-contained review for a selected ML-course lecture.

Ground course content only in public files from the selected lowercase
`lectures/<lecture_slug>/...` directory and read-only `okf/...` support.
Exclude private solutions, answer keys, grading data, teacher quiz banks,
teacher notes, untracked workbooks, URLs, other repository roots, and
case-variant lecture paths. The course wrapper verifies these rules before it
delegates generation to the portable core.

Current reviews:

| Topic | Live | Offline payload/artifact |
|---|---|---|
| Exploratory Data Analysis | [Open live review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/) | `lecture_experiences/content/lecture_01_eda.json` · `lecture_experiences/lecture_01_eda/index.html` |
| Regression | [Open live review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/) | `lecture_experiences/content/lecture_04_regression.json` · `lecture_experiences/lecture_04_regression/index.html` |
| Classification Part 1 | [Open live review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_05_classification_part_1/) | `lecture_experiences/content/lecture_05_classification_part_1.json` · `lecture_experiences/lecture_05_classification_part_1/index.html` |

Each standalone HTML file is canonical for offline and Pages use. It requires
no server, CDN, external font, account, or network request.

## Authoring and verification flow

The working pipeline is:

```text
named sources → context profile → experience specification → grounded JSON
→ deterministic generation → offline validation → browser review
→ optional publication
```

For an ML-course lecture, create the JSON payload using the portable
`references/content-contract.md`, then generate and validate it:

```powershell
$lectureSlug = 'lecture_02_data_preparation_part_1'
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py `
  --lecture-slug $lectureSlug `
  --content "lecture_experiences/content/$lectureSlug.json" `
  --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
  --output "lecture_experiences/$lectureSlug/index.html"

uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  "lecture_experiences/$lectureSlug/index.html"
```

Generation must be deterministic: regenerate into a temporary location and
compare its bytes with the committed artifact. Then review the page in a
browser at desktop and narrow mobile widths, including keyboard navigation,
visible focus, reduced motion, color-blind-safe cues, focus mode, static
fallbacks, and unavailable storage.

Quiz behavior is part of the contract. A wrong answer remains on the current
question and keeps Check answer available. A correct answer completes the
question and reveals Next question. The whole-quiz Retry keeps the four learner
settings but resets question position, attempts, completion, feedback,
results, progress, disabled inputs, and button visibility.

To publish all verified artifacts through the textbook preview:

```powershell
uv run python tools/build_textbook_preview.py
```

The builder discovers each matching
`lecture_experiences/content/<slug>.json` and
`lecture_experiences/<slug>/index.html` pair, copies the standalone artifact
byte-for-byte to `site/_build/demos/<slug>/index.html`, and exposes it on the
textbook homepage and every sidebar. Generated files under `site/_build/` are
not committed.

The repository copy of the portable core is canonical. If a global copy is
needed for unrelated repositories, synchronize it from
`.agents/skills/interactive-learning-experience-builder/` and compare the
skill, template, quiz state machine, generator, validator, and references so
drift remains visible.
