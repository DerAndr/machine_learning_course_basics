# Interactive Lecture Learning Assistant

The learning assistant creates small, self-contained lecture review pages with
grounded explanations, interactive graphs, accessibility controls, funny
topic-related break prompts, and a 10-question quiz selected from one of three
difficulty banks.

## Try the EDA Example

- [Open the live EDA review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/)
- Offline file: `lecture_experiences/lecture_01_eda/index.html`
- Editable payload: `lecture_experiences/content/lecture_01_eda.json`

Download or clone the repository and open the offline file directly. It uses no
server, CDN, external font, account, or network request.

## Learner Controls

- Foundations, Applied, or Challenge question depth
- focus-friendly (ADHD-friendly) presentation
- color-blind-safe palette with non-color visual cues
- optional funny lecture-related break prompts
- interactive histogram, box plot, scatter plot, and missingness views
- immediate quiz feedback, answer review, progress, and retry
- keyboard navigation, visible focus, reduced motion, and static fallbacks

## Workflow: portable core and course adapter

Use `.agents/skills/interactive-learning-experience-builder/SKILL.md` for the
portable content contract, deterministic single-file generator, and offline
accessibility validator. The ML-course adapter at
`.agents/skills/ml-course-interactive-learning-assistant/SKILL.md` adds only
the course-specific source hierarchy, public-safety rules, output locations,
and Pages convention. It does not copy the core template or scripts.

Before authoring a payload, record a short context profile: learner and
learning goal; authoritative and supporting sources; excluded/private sources;
requested depth and accessibility settings; output location; and available
validation, browser-testing, and publishing commands. A repository with no
`AGENTS.md`, build tool, or publishing configuration can still use the core by
recording its knowledge root and stable source identifiers. Create a thin
repository adapter only when the same local rules recur; one-off topics need
only the profile and core workflow.

## Generate Another Lecture Review

Ground content only in public course files, keep OKF read-only, and write a
payload following the portable core's
`references/content-contract.md`. The EDA payload uses the domain-neutral
`meta.experience_id`; its course slug remains its directory name.

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

## Publish Through the Textbook Preview

```powershell
uv run python tools/build_textbook_preview.py
```

The builder copies each standalone lecture review, such as
`lecture_experiences/lecture_01_eda/index.html`, to its matching Pages route,
such as `site/_build/demos/lecture_01_eda/index.html`. The standalone file
remains the canonical source; generated files under `site/_build/` are not
committed. The repository copy of the portable core is canonical too. If it is
installed globally for use in unrelated repositories, synchronize the global
copy from `.agents/skills/interactive-learning-experience-builder/` and compare
the skill, template, quiz state machine, generator, validator, and references
before use so drift is visible.

## Quiz retry behavior

A wrong answer stays on the current question, increments its attempt count,
and keeps Check answer available. Only a correct answer marks the question
complete, disables its inputs, and reveals Next question. The whole-quiz Retry preserves
the four learner settings but resets the question index, attempts,
first-attempt flags, completion flags, feedback, results, progress, disabled
inputs, and Check/Next visibility.

## Public-Safety Contract

Do not use private solutions, answer keys, grading data, teacher quiz banks,
teacher notes, or untracked workbooks. Every cited source must exist, and
lecture paths must belong to the selected lecture. The ML-course generation
wrapper enforces these path rules before delegating to the portable generator;
`okf/` paths remain allowed only as read-only supporting sources. Course
payloads may cite only canonical lowercase
`lectures/<selected_lecture_slug>/...` and `okf/...` repository paths; the
wrapper rejects every other relative root, URL, knowledge-base identifier, and
case-variant root.
