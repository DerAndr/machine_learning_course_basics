# Interactive Learning Assistant Baseline

## Prompt

```text
Create a fully offline EDA lecture review site with explanations, interactive
graphs, three difficulty levels of ten questions, focus-friendly controls,
color-blind support, funny EDA break prompts, and accessible fallbacks. Use
only public course sources and do not modify OKF.
```

## Observed response

This baseline was generated before the
`ml-course-interactive-learning-assistant` skill existed. Relevant verbatim
excerpts include:

> “I’ll treat the public Lecture 01 materials—its README, lecture notes,
> slides, examples, practical, and already-public site assets/data—as the
> content boundary”

> “Provide exactly 30 questions: ten foundational, ten intermediate, and ten
> challenge questions.”

> “Create a small, self-contained site using semantic HTML, CSS, and local
> JavaScript only”

> “Progress will remain local to the browser and the site will still function
> if storage is unavailable.”

> “Build accessible fallbacks into every graph: a meaningful title and
> description, an adjacent data table, a concise text interpretation”

> “Verify the result with the network disabled, at multiple viewport sizes,
> entirely by keyboard, with reduced-motion and high-contrast settings, and
> with automated HTML/accessibility checks plus targeted interaction tests.”

## Checklist

| Required behavior | Result | Evidence and assessment |
|---|---|---|
| Source grounding | Pass | The response names the public Lecture 01 README, lecture notes, slides, examples, practical, and existing public assets/data as its content boundary and promises source notes. |
| Exact quiz-bank counts | Pass | It specifies exactly 30 questions, divided into ten foundational, ten intermediate, and ten challenge questions. |
| One-file offline behavior | **Fail** | The response proposes a “self-contained site” but also describes local JavaScript, reviewed datasets bundled locally, and possibly an already-vendored dependency. It never requires exactly one independently portable `index.html`, embeds every runtime dependency, or defines a deterministic generator contract. |
| Chart fallbacks | Pass | It requires a title and description, adjacent data table, text interpretation, keyboard controls, and readable core content when JavaScript is unavailable. |
| Settings persistence fallback | Pass | It keeps progress local and explicitly says the site works when storage is unavailable; preferences do not depend on cookies or a backend. |
| Validation evidence | Pass | It defines offline, viewport, keyboard, reduced-motion, high-contrast, automated accessibility, and interaction checks, and promises to report the exact checks run. This is a validation-evidence contract, not a claim that the not-yet-built site already passed. |

The single observed contract gap is the absence of an explicit deterministic,
single-file output requirement. A future skill response must bind generation
to exactly one portable HTML file that opens through `file://` with no runtime
dependencies.

## Forward test after skill creation

The same scenario was run again with the completed skill and only the
skill-routed files available to the evaluator.

| Required behavior | Result | Evidence and assessment |
|---|---|---|
| Source grounding | Pass | The workflow prioritizes public Lecture 01 notes, README/metadata, examples, and student practicals while excluding solutions and grading material. |
| Exact quiz-bank counts | Pass | The output contract requires Foundations, Applied, and Challenge banks with exactly ten questions in each bank. |
| One-file offline behavior | Pass | The workflow requires one deterministic `index.html` containing HTML, CSS, JavaScript, SVG, content, data, fallbacks, and quizzes, with no server or network dependency. |
| Chart fallbacks | Pass | It requires accessible labels, non-color cues, live summaries, and static fallbacks. |
| Settings persistence fallback | Pass | All settings remain changeable in-page, and storage failure must not break the experience. |
| OKF preservation | Pass | OKF is explicitly read-only and must not be modified. |
| Validation evidence | Pass | The evaluator identified both generator and validator commands plus a `file://` interaction and console-cleanliness smoke test. |

The skill closes the baseline's single-file contract gap. The evaluator noted
that initial settings and output path may be unspecified in a fresh prompt;
the skill handles these as generation defaults while keeping every setting
changeable in the generated page.
