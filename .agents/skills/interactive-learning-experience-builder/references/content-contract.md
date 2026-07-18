# Content contract

Write one UTF-8 JSON object with exactly these top-level keys:

```text
meta, defaults, concepts, visualizations, quizzes, break_prompts
```

- `meta`: Include non-empty `experience_id`, `title`, and a non-empty `sources`
  array. A source is a safe repository-relative path, an `http://` or
  `https://` provenance URL, or an explicit identifier matching
  `^[A-Za-z][A-Za-z0-9+.-]*:[^/].+`. Repository-relative paths are checked for
  existence when a repository root is supplied; URLs and identifiers are not.
- `defaults`: Include `difficulty` (`foundations`, `applied`, or `challenge`)
  plus boolean `focus_mode`, `color_blind`, and `break_prompts` values.
- `concepts`: Include stable `id`, `title`, `explanation`, `interpretation`,
  non-empty `common_mistakes`, and named `sources`.
- `visualizations`: Include stable `id`, supported `type`, `title`,
  `explanation`, embedded `data`, and a readable `fallback`.
- `quizzes`: Include `foundations`, `applied`, and `challenge` arrays. Each
  contains exactly 10 question objects with stable IDs, a supported response
  type, prompt, options, answer, explanation, and assessed concept ID.
- `break_prompts`: Always embed at least one readable prompt. The default only
  controls the initial display state.

The current reusable template supports `histogram`, `boxplot`, `scatter`, and
`missingness` visualizations. Use labels, shapes, patterns, or line styles in
addition to color, and ensure the fallback communicates the essential lesson.

Keep every concept explanation and quiz prompt, option, answer, and explanation
in the static representation so learners retain access without JavaScript.
