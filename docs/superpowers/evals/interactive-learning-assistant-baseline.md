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
