# Student Learning Companion Quickstart

Use this guide when you want a short interactive review instead of navigating
an entire lecture or knowledge repository. Start with an existing review when
one covers your topic. Ask Codex to generate a new one only when you need a
different topic or learning goal.

## Pick the shortest path

| What you need | What to do | Skills needed |
|---|---|---|
| A prepared ML review | Open one of the links below | None |
| Help finding the right course material | Ask Codex to use `$ml-course-student-navigator` | Repository skill |
| A new review from this course | Open this repository in Codex and use both learning-experience skills | Repository skills |
| A review from another repository or knowledge base | Add the generic skill to that project or your personal Codex | Generic skill |

Prepared reviews:

- [Exploratory Data Analysis](https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/)
- [Regression](https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/)
- [Classification Part 1](https://derandr.github.io/machine_learning_course_basics/demos/lecture_05_classification_part_1/)

These pages work without Codex. A cloned repository also contains each
standalone review under `lecture_experiences/<lecture_slug>/index.html`.

## Use the skills from this repository

Codex reads repository skills from `.agents/skills`. You do not need to copy or
install this course's skills when you work inside a clone of this repository.

1. Clone the repository and enter its root:

   ```powershell
   git clone https://github.com/DerAndr/machine_learning_course_basics.git
   cd machine_learning_course_basics
   ```

2. Open that folder in the Codex desktop app, IDE extension, or start Codex CLI
   from the repository root.
3. In CLI or IDE, run `/skills` or type `$` in the prompt to check that these
   skills are visible:

   - `$interactive-learning-experience-builder` — the portable, generic core;
   - `$ml-course-interactive-learning-assistant` — this course's source and
     publishing rules;
   - `$ml-course-student-navigator` — help finding existing course materials.

4. Paste one of the prompts below. Codex may ask for permission before writing
   generated files or running validation.

If you only want help finding an existing resource, use:

> Use $ml-course-student-navigator. I want to review classification metrics in
> about 15 minutes. Point me to the shortest suitable existing review or course
> material. Do not generate new files.

## Course-specific prompt examples

### Create a focused Regression review

Copy this prompt and change the topic, lecture slug, duration, or focus:

```text
Use $ml-course-interactive-learning-assistant with
$interactive-learning-experience-builder.

Create a 15-minute interactive review from lecture_04_regression.
Learner and goal: a university student revising before an exam.
Focus: residuals, regularization, feature scaling, and choosing a regression
metric.
Topic-specific interaction: include a control that changes one
topic-relevant interpretation, such as the residual pattern, Ridge/Lasso
strength, or MAE/RMSE sensitivity.
Accessibility defaults: focus-friendly mode on, color-blind-safe mode on,
reduced motion respected, and break prompts off.
Source rule: use only public course sources allowed by the ML-course adapter.

First show me the context profile and source plan. Then create the grounded
content payload and offline HTML, run the repository validator, and tell me the
exact local file to open. Exercise every interaction and test the
color-blind-safe setting before reporting completion. Do not publish anything.
```

### Get a personal explanation without generating a page

```text
Use $ml-course-student-navigator with
$ml-course-interactive-learning-assistant.

I am confused about precision, recall, and threshold choice in
lecture_05_classification_part_1. Teach me through five short questions,
one at a time. Use the public lecture sources, give feedback after every
answer, and do not create or change files.
```

The second prompt is cheaper and faster when you need a temporary tutoring
conversation rather than a reusable webpage.

## Use the generic skill in another repository

The portable skill works with named, trustworthy sources in a different
repository or knowledge base. Open that project in Codex and use this template:

```text
Use $interactive-learning-experience-builder.

Create a short, grounded, offline interactive learning companion.
Learner and goal: [who will use it and what they should be able to do].
Topic: [one focused topic].
Trusted sources: [exact repository paths, approved URLs, or knowledge-base
identifiers].
Excluded material: [private, draft, unrelated, or unsafe sources].
Duration and level: [for example, 15 minutes at beginner level].
Accessibility defaults: [focus, color, motion, and break preferences].
Topic-specific interaction: [one control that must change an interpretation
that matters for this topic, not only the chart appearance].
Output path: [where the standalone HTML should be written].
Available validation command: [command, or say none].

First show the context profile and source plan. Ask before changing the source
scope. Then create the grounded payload, generate the offline experience, run
available validation, exercise the topic-specific interaction, test the
color-blind-safe setting, and report the output path.
```

For example, in a software project:

```text
Use $interactive-learning-experience-builder.

Create a 12-minute beginner learning companion about contributing a safe pull
request to this project.
Learner and goal: a first-time contributor who should be able to prepare and
submit a small documentation change.
Trusted sources: README.md, CONTRIBUTING.md, and docs/review-process.md.
Excluded material: secrets, private issue links, generated files, and source
files outside those named paths.
Accessibility defaults: reduced motion and color-blind-safe cues.
Topic-specific interaction: let the learner choose a pull-request scenario and
show how that choice changes the recommended safety checks.
Output path: learning_experiences/first-contribution/index.html.

Show the context profile first. Then generate and validate a self-contained
offline page. Test every interaction in both palette modes. Do not publish or
open a pull request.
```

Keep the scope to one topic and name each trusted source explicitly. This
reduces token use and makes grounding easier to verify. A useful interaction
must change the learner's interpretation of the topic; a generic animation or
color change is not enough.

## Add the generic skill to your personal Codex

Personal installation makes the generic skill available when you open other
projects. Copy the complete directory—including `SKILL.md`, `assets/`,
`references/`, and `scripts/`—not only the `SKILL.md` file.

From the root of this cloned repository, use one of the following commands.

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.agents\skills" | Out-Null
Copy-Item -Recurse -Force `
  ".agents\skills\interactive-learning-experience-builder" `
  "$HOME\.agents\skills\"
```

macOS or Linux:

```bash
mkdir -p "$HOME/.agents/skills"
cp -R .agents/skills/interactive-learning-experience-builder \
  "$HOME/.agents/skills/"
```

Then open another repository in Codex and invoke
`$interactive-learning-experience-builder`. Codex normally detects skill
changes automatically; restart Codex if the skill does not appear.

Install only the generic skill globally. Keep
`$ml-course-interactive-learning-assistant` in this repository because it
depends on this course's source hierarchy, exclusions, output locations, and
publishing rules.

For installation locations, invocation behavior, and plugin distribution, see
the [official OpenAI Codex skill documentation](https://learn.chatgpt.com/docs/build-skills).
For wider one-click distribution, the next step would be packaging the generic
skill as a plugin; manual installation is enough for local use and testing.

## Troubleshooting

| Problem | Check |
|---|---|
| The skill is not listed | Open Codex at the repository root, confirm the skill folder directly contains `SKILL.md`, then restart Codex |
| Codex uses the wrong material | Name the trusted paths and excluded material explicitly |
| You only need a quick answer | Ask for a short tutoring conversation and say `do not create or change files` |
| Generation or validation cannot run | Complete the repository setup with `uv sync`, then retry |
| The generated page is hard to find | Ask Codex to report the exact absolute output path |
| The request is becoming expensive | Use one topic, one learner goal, named sources, and a 10–15 minute target |

Students should not include private notes, answer keys, grading data, secrets,
or other restricted material in the source list. For the full deterministic
generation and publishing workflow, use the
[operational learning-companion guide](interactive-lecture-learning-assistant.md).
