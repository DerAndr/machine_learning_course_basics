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

## Generate Another Lecture Review

Use `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md`. Ground
content only in public course files, keep OKF read-only, and write a payload
following the skill's `references/content-contract.md`.

```powershell
$lectureSlug = 'lecture_02_data_preparation_part_1'
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_lecture_site.py `
  --content "lecture_experiences/content/$lectureSlug.json" `
  --template .agents/skills/ml-course-interactive-learning-assistant/assets/lecture-site-template.html `
  --output "lecture_experiences/$lectureSlug/index.html"

uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/validate_lecture_site.py `
  "lecture_experiences/$lectureSlug/index.html"
```

## Publish Through the Textbook Preview

```powershell
uv run python tools/build_textbook_preview.py
```

The builder copies each standalone lecture review, such as
`lecture_experiences/lecture_01_eda/index.html`, to its matching Pages route,
such as `site/_build/demos/lecture_01_eda/index.html`. The standalone file
remains the source; generated files under `site/_build/` are not committed.

## Public-Safety Contract

Do not use private solutions, answer keys, grading data, teacher quiz banks,
teacher notes, or untracked workbooks. Every cited source must exist, and
lecture paths must belong to the selected lecture.
