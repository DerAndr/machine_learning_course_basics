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
  type, prompt, options, answer, explanation, and assessed concept ID. Every
  option must be a readable string. Multiple-choice answers must contain unique
  choices from their options. An interpretation question with options is
  rendered as a choice question, so its answer must be one of those options;
  use an empty options array only for a free-text interpretation.
- `break_prompts`: Always embed at least one readable prompt. The default only
  controls the initial display state.

## Visualization type selection

Choose the type that best teaches the named learning objective. Keep generic
chart types when the goal is to read that graphical form; use a semantic type
when its data model and control express the actual decision, diagnostic, or
comparison. Preserve every meaningful payload field in the rendering and
summary. Use labels, shapes, patterns, or line styles in addition to color, and
ensure the fallback communicates the essential lesson.

| Type | Learning purpose | Required data shape | Control shape |
| --- | --- | --- | --- |
| `histogram` | Compare a numeric distribution under binning choices. | Numeric array. | `bins` array. |
| `boxplot` | Inspect centre, spread, and IQR outlier fences. | Numeric array. | `fence_multipliers` array. |
| `scatter` | Inspect association between two numeric variables. | Array of `{x, y}` points. | `trend_line` boolean. |
| `missingness` | Compare missing-data proportions with denominators. | Array of `{label, missing, total}` rows. | `sort` choices. |
| `binary-threshold` | Connect scores, a decision threshold, confusion outcomes, precision, and recall. | At least four `{id, score, actual}` records plus positive/negative labels. | Numeric threshold range: `minimum`, `maximum`, `step`, `initial`. |
| `labeled-scatter` | Compare meaningful groups against candidate linear boundaries. | At least four `{id, x, y, series}` points plus two series labels. | Boundary choices with `id`, `label`, `slope`, `intercept`, and `initial`. |
| `residual-diagnostics` | Compare observed/fitted values and residual patterns across scenarios. | `scenarios` with labeled `{id, x, observed, predicted}` points. | Initial scenario ID. |
| `coefficient-path` | Compare Ridge shrinkage with Lasso sparsity as penalty changes. | Increasing penalties and feature `ridge`/`lasso` paths. | Penalty `initial_index`. |
| `error-metrics` | Compare MAE, MSE, and RMSE as one error grows. | Fixed `base_errors`, increasing `adjustable_error`, and units. | Adjustable-error `initial_index`. |

## Semantic visualization validation boundaries

Use the following five schemas exactly. The generator rejects a payload outside
these boundaries before it produces HTML.

### `binary-threshold`

Required fields: `data` records with `id`, `score`, and `actual`; `controls`
with `minimum`, `maximum`, `step`, and `initial`; and `labels` with `positive`
and `negative`.

Validation rules:

- at least four records;
- unique non-empty IDs;
- finite scores from zero through one;
- `actual` is exactly `0` or `1`;
- finite control values satisfying `0 <= minimum <= initial <= maximum <= 1` and `minimum < maximum`;
- positive `step` no larger than `maximum - minimum`;
- non-empty positive and negative labels.

### `labeled-scatter`

Required fields: `data` points with `id`, `x`, `y`, and `series`; `controls`
with `boundaries` and `initial`; and `labels` with `x_axis`, `y_axis`,
`series`, and `positive_series`.

Validation rules:

- at least four points and two non-empty series;
- unique point IDs and finite coordinates;
- every point series exists in `labels.series`;
- exactly two supported series for the first implementation;
- unique boundary IDs with finite slopes and intercepts;
- the initial boundary exists;
- the positive series exists;
- non-empty axis and display labels.

### `residual-diagnostics`

Required fields: `data.scenarios` with labeled scenarios and points containing
`id`, `x`, `observed`, and `predicted`; `controls.initial`; and `x_axis`,
`target_axis`, and `residual_axis` labels.

Validation rules:

- at least one scenario;
- unique scenario and point IDs;
- at least five finite points per scenario;
- the initial scenario exists;
- non-empty axis labels.

### `coefficient-path`

Required fields: `data.penalties`, `data.series` entries with `feature`,
`ridge`, and `lasso`, plus `controls.initial_index`.

Validation rules:

- at least three increasing, finite, non-negative penalties;
- at least two feature series with unique non-empty names;
- Ridge and Lasso arrays match the penalty-array length;
- all coefficients are finite;
- the initial index is in range.

### `error-metrics`

Required fields: `data.base_errors`, `data.adjustable_error`,
`controls.initial_index`, and `labels.units`.

Validation rules:

- at least three finite base errors;
- at least three finite, non-negative adjustable-error values in increasing order;
- the initial index is in range;
- a non-empty units label.

Keep every concept explanation and quiz prompt, option, answer, and explanation
in the static representation so learners retain access without JavaScript.
