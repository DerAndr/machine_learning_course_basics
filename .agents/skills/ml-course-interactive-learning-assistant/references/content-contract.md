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
  set the other values to booleans. Treat `break_prompts` as the initial display
  state, not as permission to omit prompt content.
- `concepts`: Provide grounded concept objects with a stable `id`, `title`,
  concise `explanation`, interpretation guidance, common mistakes, and source
  references.
- `visualizations`: Provide visualization objects as defined below.
- `quizzes`: Include `foundations`, `applied`, and `challenge` arrays. Each array
  must contain exactly 10 question objects.
- `break_prompts`: Always embed `break_prompts` content as short, funny,
  lecture-related strings. Its generation default controls only the initial state
  so the learner can enable prompts inside the finished page.

Keep explanations and full quiz content statically readable if JavaScript fails.
Include every concept explanation and every quiz prompt, option, answer, and
explanation in that static representation.

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

For `single-choice`, make `answer` exactly one value from `options`. For
`multiple-choice`, make it a non-empty list containing only values from
`options`. For `interpretation`, use a non-empty expected response string.

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
Make functioning charts encode meaning with labels, shapes, patterns, or line styles in addition to color.

Use these exact data shapes:

| Type | `data` | Interactive control |
|---|---|---|
| `histogram` | Non-empty array of finite numbers. | `controls.bins`: positive integer choices from 1 to 50; defaults to `4, 6, 8`. |
| `boxplot` | Non-empty array of finite numbers. | Whisker multiplier choices are supplied by the template. |
| `scatter` | Non-empty array of objects with finite `x` and `y`. | Learner toggles a labeled trend line. |
| `missingness` | Non-empty array of `{label, missing, total}` where `total > 0` and `0 <= missing <= total`. | Learner switches between source and descending order. |
