# Content Contract

Write one UTF-8 JSON object with exactly these top-level keys:

```text
meta, defaults, concepts, visualizations, quizzes, break_prompts
```

## Required shape

- `meta`: Include the lecture slug, title, and `sources`. List repository-relative
  public course files that ground the payload.
- `defaults`: Include `difficulty`, `focus_mode`, `color_blind`, and
  `break_prompts`. Set `difficulty` to `foundations`, `applied`, or `challenge`;
  set the other values to booleans.
- `concepts`: Provide grounded concept objects with a stable `id`, `title`,
  concise `explanation`, interpretation guidance, common mistakes, and source
  references.
- `visualizations`: Provide visualization objects as defined below.
- `quizzes`: Include `foundations`, `applied`, and `challenge` arrays. Each array
  must contain exactly 10 question objects.
- `break_prompts`: Provide short, funny, lecture-related strings. Use an empty
  array when break prompts are disabled.

## Quiz questions

Give every question these fields:

| Field | Requirement |
|---|---|
| `id` | Stable and unique across all quiz banks. |
| `type` | Supported response format such as `single-choice`, `multiple-choice`, or `interpretation`. |
| `prompt` | Grounded, unambiguous question text. |
| `options` | Array of answer choices; use an empty array only when the response type does not use choices. |
| `answer` | Correct option value, list of values, or expected interpretation supported by the source. |
| `explanation` | Concise feedback that explains why the answer is correct. |
| `concept` | ID of the concept assessed. |

Do not derive questions from private solutions, answer keys, grading data, or
untracked quiz workbooks.

## Visualizations

Set each visualization `type` to one of:

```text
histogram, boxplot, scatter, missingness
```

Include a stable `id`, title, explanatory text, embedded data, and any controls
needed by the selected type. Include `fallback` on every visualization. Make
`fallback` a readable text summary and/or data table that communicates the same
essential lesson without JavaScript, SVG interaction, or color alone.
