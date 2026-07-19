# Semantic Learning Visualizations Design

## Purpose

Improve the portable learning-experience system so generated visualizations
teach topic-specific ideas instead of reusing superficially different generic
charts. Make the color-blind-safe setting produce a clearly visible change in
graph marks while retaining labels, shapes, patterns, and text fallbacks.

The change must strengthen both the current ML-course showcase and every future
experience generated with the portable core.

## Problem statement

Visual review of the published Regression and Classification companions found
two related issues.

First, both experiences use the same scatter–histogram–boxplot combination.
Their titles and numeric values differ, but their visual grammar, controls, and
interaction pattern are nearly identical. A student can reasonably conclude
that the experiences are generic chart demonstrations rather than
topic-specific learning tools.

Second, the Classification scatter payload contains `class: "A"` and
`class: "B"` values, but the shared renderer ignores them. It assigns circles
and squares by point index. The graph therefore fails to encode the class
separation it claims to explain.

The color-blind-safe setting is connected to the CSS variables and changes the
computed graph colors. However, the standard palette and safe palette are both
blue/orange pairs:

- standard: `#315c9b` and `#c55a11`;
- color-blind-safe: `#0072b2` and `#d55e00`.

The difference is technically present but perceptually weak. Existing tests
check for the setting hook and non-color cues, not for a meaningful palette
change or semantic class encoding.

## Goals

- Give Classification a threshold/confusion-matrix explorer and a true
  class-aware decision-boundary explorer.
- Give Regression three distinct interactions: fit/residual diagnostics,
  regularization behavior, and metric sensitivity to an extreme error.
- Make the color-blind-safe setting visibly change graph colors in EDA,
  Regression, Classification, and future generated experiences.
- Preserve shape, pattern, label, line-style, summary, and static fallback
  cues so color is never the only signal.
- Extend the portable core, source contract, validation, skill instructions,
  course adapter, source payloads, generated artifacts, tests, and
  documentation together.
- Keep generation deterministic and every experience self-contained,
  offline-capable, and usable through `file://`.
- Use only the existing public lecture sources permitted by the ML-course
  adapter.
- Strengthen the Build Week submission narrative with an evidence-backed
  feedback and iteration story.

## Non-goals

- Do not hand-edit generated `lecture_experiences/<slug>/index.html` files.
- Do not introduce a charting dependency, CDN, web font, server, account, or
  network request.
- Do not create arbitrary adapter-provided JavaScript callbacks.
- Do not redesign the quiz state machine or change the 10-question banks.
- Do not add new lecture topics.
- Do not package the skill as a plugin in this change.
- Do not remove existing histogram, boxplot, scatter, or missingness support.

## Architectural choice

Extend the portable core with semantic, data-driven visualization types and
regenerate every affected artifact.

This is preferred over page-specific patches because the durable behavior
belongs in the core contract and renderer. It is preferred over custom
repository callbacks because a closed, validated type set preserves
determinism, offline safety, accessibility, and testability.

## Responsibility boundaries

### Portable core

The portable `interactive-learning-experience-builder` owns:

- supported visualization types and schemas;
- schema validation and readable errors;
- pure visualization calculations;
- deterministic SVG rendering;
- palette behavior and non-color encodings;
- live summaries and static fallbacks;
- offline validation and reusable tests;
- instructions for selecting semantically appropriate interactions.

### ML-course adapter

The `ml-course-interactive-learning-assistant` owns:

- the public course source hierarchy and exclusions;
- the requirement to select visualizations that match a lecture learning
  objective;
- course output paths, generation wrapper, validation, and publication rules;
- a check against repeating a generic visualization set when a supported
  semantic interaction better represents the lesson.

### Lecture payload

Each grounded JSON payload owns:

- the illustrative data and provenance;
- meaningful axis, series, control, and scenario labels;
- initial control values;
- explanation, interpretation, and static fallback text;
- the mapping from a visualization to the assessed course concept.

The generated HTML remains derived output.

## Runtime architecture

The data flow is:

```text
named public sources
→ grounded semantic visualization payload
→ Python schema validation
→ deterministic generator
→ embedded pure visualization model
→ interactive SVG and live text summary
→ static readable fallback
```

Add
`.agents/skills/interactive-learning-experience-builder/assets/visualization-models.js`
for pure calculations shared by browser rendering and Node tests. The
generator embeds this source into the standalone HTML just as it embeds the
quiz state machine. It must not remain as an external runtime file reference.

The HTML template owns DOM construction and SVG markup. The pure model owns
classification counts, classification metrics, residual calculation,
coefficient selection, error metrics, stable series mapping, and numeric
guardrails.

## New portable visualization types

Existing `histogram`, `boxplot`, `scatter`, and `missingness` types remain
supported.

### `binary-threshold`

Purpose: connect probability scores, a decision threshold, confusion-matrix
outcomes, precision, and recall.

Required payload fields:

```json
{
  "type": "binary-threshold",
  "data": [
    {"id": "case-01", "score": 0.82, "actual": 1}
  ],
  "controls": {
    "minimum": 0.1,
    "maximum": 0.9,
    "step": 0.05,
    "initial": 0.5
  },
  "labels": {
    "positive": "Positive",
    "negative": "Negative"
  }
}
```

Validation rules:

- at least four records;
- unique non-empty IDs;
- finite scores from zero through one;
- `actual` is exactly `0` or `1`;
- finite control values satisfying
  `0 <= minimum <= initial <= maximum <= 1` and `minimum < maximum`;
- positive `step` no larger than `maximum - minimum`;
- non-empty positive and negative labels.

Runtime behavior:

- a range control changes the threshold;
- prediction is positive when `score >= threshold`;
- TP, FP, TN, and FN are recomputed;
- precision and recall show a numeric value when defined and `not defined`
  when their denominator is zero;
- a labeled 2×2 matrix uses text, patterns, and outcome abbreviations in
  addition to color;
- the live summary states the threshold, counts, precision, and recall.

### `labeled-scatter`

Purpose: show meaningful groups and compare candidate linear decision
boundaries.

Required payload fields:

```json
{
  "type": "labeled-scatter",
  "data": [
    {"id": "point-01", "x": 1.0, "y": 1.2, "series": "A"}
  ],
  "controls": {
    "boundaries": [
      {
        "id": "balanced",
        "label": "Balanced boundary",
        "slope": 1.0,
        "intercept": 0.0
      }
    ],
    "initial": "balanced"
  },
  "labels": {
    "x_axis": "Feature 1",
    "y_axis": "Feature 2",
    "series": {"A": "Class A", "B": "Class B"},
    "positive_series": "B"
  }
}
```

Validation rules:

- at least four points and two non-empty series;
- unique point IDs and finite coordinates;
- every point series exists in `labels.series`;
- exactly two supported series for the first implementation;
- unique boundary IDs with finite slopes and intercepts;
- the initial boundary exists;
- the positive series exists;
- non-empty axis and display labels.

Runtime behavior:

- series, not point position, determines color, shape, and accessible label;
- a selector changes the displayed candidate boundary;
- each point is labeled with coordinates, true series, and boundary-side
  prediction;
- the summary reports correct and incorrect side counts for the illustrative
  boundary;
- the boundary uses a line style that remains visible in either palette.

### `residual-diagnostics`

Purpose: connect observed values, fitted values, residual signs, curvature,
and changing variance.

Required payload fields:

```json
{
  "type": "residual-diagnostics",
  "data": {
    "scenarios": [
      {
        "id": "appropriate",
        "label": "Appropriate linear fit",
        "points": [
          {"id": "obs-01", "x": 1.0, "observed": 2.1, "predicted": 2.0}
        ]
      }
    ]
  },
  "controls": {"initial": "appropriate"},
  "labels": {
    "x_axis": "Predictor",
    "target_axis": "Observed target",
    "residual_axis": "Residual"
  }
}
```

Validation rules:

- at least one scenario;
- unique scenario and point IDs;
- at least five finite points per scenario;
- the initial scenario exists;
- non-empty axis labels.

Runtime behavior:

- a selector switches between the three fixed, grounded illustrative
  scenarios;
- the upper plot shows observed points and their fitted relationship;
- the lower plot shows `observed - predicted` against fitted value;
- a zero residual reference and residual sign labels are always visible;
- the summary describes random scatter, curvature, or changing spread without
  claiming a statistical test.

### `coefficient-path`

Purpose: compare Ridge shrinkage with Lasso sparsity as penalty strength
changes.

Required payload fields:

```json
{
  "type": "coefficient-path",
  "data": {
    "penalties": [0.0, 0.25, 0.5, 1.0],
    "series": [
      {
        "feature": "Feature A",
        "ridge": [2.8, 2.3, 1.8, 1.2],
        "lasso": [2.8, 2.1, 1.1, 0.0]
      }
    ]
  },
  "controls": {"initial_index": 0}
}
```

Validation rules:

- at least three increasing, finite, non-negative penalties;
- at least two feature series with unique non-empty names;
- Ridge and Lasso arrays match the penalty-array length;
- all coefficients are finite;
- the initial index is in range.

Runtime behavior:

- a penalty slider selects one precomputed step;
- coefficient paths remain visible while current values are emphasized;
- the current Ridge and Lasso values are listed as text;
- exact Lasso zeros use a distinct zero marker and label;
- the summary distinguishes shrinkage from sparsity without suggesting that
  feature selection proves causality.

### `error-metrics`

Purpose: show why MAE and RMSE react differently to a large error.

Required payload fields:

```json
{
  "type": "error-metrics",
  "data": {
    "base_errors": [-2, -1, 0, 1, 2],
    "adjustable_error": [0, 5, 10, 20]
  },
  "controls": {"initial_index": 1},
  "labels": {"units": "target units"}
}
```

Validation rules:

- at least three finite base errors;
- at least three finite, non-negative adjustable-error values in increasing
  order;
- the initial index is in range;
- a non-empty units label.

Runtime behavior:

- a slider changes the single adjustable error;
- MAE, MSE, and RMSE are recomputed with that error included;
- the extreme error uses a labeled shape distinct from base errors;
- the summary reports all metrics in the appropriate units and explains that
  squaring gives the large miss more influence.

## Palette and non-color behavior

The standard graph palette becomes:

- primary: purple `#6d28d9`;
- secondary: teal `#0f766e`.

The color-blind-safe graph palette becomes:

- primary: blue `#0072b2`;
- secondary: vermillion `#d55e00`.

Graph-specific CSS variables must be separate from general button and link
colors so the setting visibly affects marks, lines, matrix cells, boundaries,
and coefficient paths. The page displays a short live status:
`Palette: standard` or `Palette: color-blind-safe`.

Every series or outcome also uses at least one non-color cue:

- circle versus square or triangle;
- solid versus hatched fill;
- solid versus dashed line;
- direct text label or abbreviation;
- a live textual summary;
- a complete static fallback.

Palette changes must not alter data, control state, quiz state, or the semantic
meaning of a mark.

## Course payload redesign

### Classification Part 1

Replace the generic visualization trio with:

1. `binary-threshold` using a fixed illustrative set of scores and actual
   outcomes grounded in the lecture treatment of logistic probabilities,
   thresholds, confusion matrices, precision, and recall.
2. `labeled-scatter` using two semantically encoded classes and multiple
   illustrative boundaries grounded in the lecture treatment of decision
   geometry.

The threshold view is the primary interaction. The boundary view is the second
interaction. Neither uses index-based styling.

### Regression

Replace the generic visualization trio with:

1. `residual-diagnostics` covering scenarios with the IDs `appropriate`,
   `curvature`, and `funnel`;
2. `coefficient-path` comparing Ridge and Lasso at fixed penalty strengths;
3. `error-metrics` comparing MAE and RMSE as one error grows.

The illustrative values remain pedagogical fixtures rather than claims about
a real fitted dataset. Explanations and fallbacks must say so.

### Exploratory Data Analysis

Keep histogram, boxplot, scatter, and missingness interactions because they
directly match EDA learning goals. Regenerate the page with the new palette
variables, visible palette status, and any shared accessibility improvements.

## Skill and contract changes

Update the portable skill to require:

- at least one visualization whose control changes a topic-relevant
  interpretation, not only presentation;
- semantic payload fields to be preserved by the renderer;
- meaningful axis, series, scenario, and control labels;
- a visibly distinct palette change;
- non-color cues and equivalent live/static summaries;
- browser exercise of every visualization control and palette mode.

Update the content contract with the five exact schemas and selection guidance.
Generic histogram, boxplot, scatter, and missingness types remain appropriate
when the learning goal is about those graphical forms.

Update the ML-course adapter to require matching each visualization to a named
lecture objective and to reject an unexplained repeated generic chart set when
a semantic type is supported.

## Validation and error handling

Python payload validation fails before generation when any new schema is
invalid. Error messages identify the visualization index, type, and field.

Pure runtime calculations handle:

- zero precision or recall denominators;
- threshold extremes;
- constant coordinate ranges;
- empty predicted groups;
- zero residual ranges;
- zero MAE or RMSE;
- coefficient values equal to zero;
- unavailable local storage.

Invalid payloads do not produce partial HTML. Runtime controls operate only on
validated embedded content. Static content remains readable when JavaScript is
unavailable.

## Test strategy

Implementation follows red–green–refactor.

### Python contract and generator tests

Extend `tests/test_lecture_site_generator.py` to cover:

- acceptance of every valid new schema;
- focused rejection tests for each validation rule;
- visualization-model script embedding;
- palette status and graph-variable hooks;
- deterministic generation;
- unchanged support for existing visualization types;
- no external runtime resources.

### Pure visualization-model tests

Add `tests/visualization_models.test.js` to verify:

- TP, FP, TN, and FN at low, middle, and high thresholds;
- precision and recall including zero-denominator cases;
- stable class-to-shape mapping independent of point order;
- boundary-side predictions;
- residual sign and scenario summaries;
- Ridge and Lasso value selection including exact Lasso zeros;
- MAE, MSE, and RMSE with and without the extreme error.

### Experience-specific tests

Update:

- `tests/test_classification_part_1_lecture_experience.py` to require
  `binary-threshold` and `labeled-scatter`, valid class semantics, correct
  source grounding, and deterministic regeneration;
- `tests/test_regression_lecture_experience.py` to require all three Regression
  types, valid scenarios and paths, correct source grounding, and deterministic
  regeneration;
- `tests/test_eda_lecture_experience.py` to require the shared palette status
  and preserve the current EDA visualization set.

### Skill and documentation tests

Update:

- `tests/test_interactive_learning_experience_builder_skill.py`;
- `tests/test_interactive_learning_assistant_skill.py`;
- `tests/test_interactive_learning_assistant_docs.py`;
- `tests/test_textbook_preview.py`.

These tests enforce semantic visualization guidance, documentation links,
updated submission evidence, and publication of regenerated pages.

### Browser acceptance

Open each generated page directly through `file://` and through its Pages URL.
At desktop and narrow mobile widths:

- exercise every control;
- verify keyboard operation and visible focus;
- verify the live summary after every change;
- toggle the palette and confirm graph marks visibly change;
- confirm shapes, patterns, labels, and line styles remain sufficient;
- verify reduced motion, focus mode, static fallbacks, and unavailable storage;
- verify no horizontal page overflow.

## Documentation and submission updates

Update:

- `README.md` with the improved topic-specific interaction summary;
- `AGENTS.md` with the expanded semantic visualization workflow and checks;
- `docs/learning-companions-architecture.md` with the visualization-model
  boundary;
- `docs/interactive-lecture-learning-assistant.md` with authoring and browser
  verification instructions;
- `docs/student-learning-companion-quickstart.md` with examples requesting
  topic-specific interactions;
- `docs/build-week-integration-evidence.md` with the new implementation and
  verification evidence;
- `docs/build-week-submission-preparation.md` with revised submission fields,
  test flow, screenshots, and demo narration.

The submission story explains that visual review revealed:

1. repeated chart grammar that weakened topic identity;
2. ignored class semantics in a Classification plot;
3. a technically functioning but perceptually ineffective palette switch.

It then shows how the reusable core, not only the showcase pages, was improved.
This demonstrates a credible feedback loop: observe learner-facing behavior,
trace the issue to the shared system, improve the contract, regenerate, and
verify.

The revised demo flow includes:

1. Classification threshold movement changing the confusion matrix, precision,
   and recall;
2. Classification boundary movement with class-consistent shapes;
3. Regression residual scenario switching;
4. Regression penalty movement comparing Ridge and Lasso;
5. Regression extreme-error movement comparing MAE and RMSE;
6. a quick standard versus color-blind-safe palette toggle;
7. the generic skill and deterministic regeneration path.

## Regeneration and publication

After source, skill, template, and payload changes:

1. generate EDA, Regression, and Classification HTML through the portable core
   and ML-course wrapper;
2. validate each standalone HTML file;
3. regenerate each artifact to a temporary path and compare bytes;
4. run the complete Python and Node test suites;
5. run Ruff formatting and lint, strict OKF validation, and textbook build;
6. inspect local `file://` pages in a browser;
7. publish through the existing GitHub Pages workflow;
8. wait for GitHub Actions and inspect all live routes.

## Acceptance criteria

- Classification contains a working threshold/confusion-matrix explorer and a
  working class-aware decision-boundary explorer.
- Classification class labels determine mark shape, color, text, and summary.
- Regression contains all three approved semantic interactions.
- Standard and color-blind-safe palettes are visibly different on every page.
- Every visualization remains understandable without color and without
  JavaScript.
- EDA retains its appropriate visualization set and receives shared palette
  improvements.
- The portable skill, ML adapter, contract, generator, validator, payloads,
  artifacts, tests, repository docs, and submission docs agree.
- Generation remains deterministic and produces self-contained offline HTML.
- All local checks and GitHub Actions pass.
- The published homepage and all three review routes expose the regenerated
  experiences.
