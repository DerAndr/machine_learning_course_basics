# Semantic Learning Visualizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the portable learning-experience core with semantic Classification and Regression visualizations, a perceptibly different color-blind-safe palette, deterministic regeneration, and aligned repository and Build Week documentation.

**Architecture:** Put pure calculations in one browser-and-Node-compatible visualization model, validate a closed set of semantic payload schemas in Python, and keep SVG/DOM rendering in the self-contained HTML template. Update the ML payloads and regenerate all three companions from the core; never hand-edit generated HTML.

**Tech Stack:** Python 3.12, JavaScript on Node 22 and in-browser, deterministic HTML/SVG/CSS, pytest, Ruff, GitHub Actions, GitHub Pages

## Global Constraints

- Use only named public sources permitted by the ML-course adapter.
- Preserve `file://` operation without a server, account, CDN, font, chart dependency, or network request.
- Keep histogram, boxplot, scatter, and missingness backward compatible.
- Add exactly these semantic visualization types: `binary-threshold`, `labeled-scatter`, `residual-diagnostics`, `coefficient-path`, and `error-metrics`.
- Standard graph colors are purple `#6d28d9` and teal `#0f766e`.
- Color-blind-safe graph colors are blue `#0072b2` and vermillion `#d55e00`.
- Every color encoding also has a shape, pattern, line-style, direct-label, live-summary, or static-fallback cue.
- Classification uses `binary-threshold` and `labeled-scatter`.
- Regression uses `residual-diagnostics`, `coefficient-path`, and `error-metrics`.
- EDA retains histogram, boxplot, scatter, and missingness and receives the shared palette/runtime update.
- Quiz content and quiz state behavior do not change.
- Generated HTML is derived output and is changed only by deterministic regeneration.
- Follow red–green–refactor for every behavior change.

---

### Task 1: Pure Visualization Calculations

**Files:**
- Create: `.agents/skills/interactive-learning-experience-builder/assets/visualization-models.js`
- Create: `tests/visualization_models.test.js`

**Interfaces:**
- Consumes: validated plain JavaScript objects embedded from the payload.
- Produces: global/CommonJS object `LearningVisualizationModels` with `thresholdSummary`, `seriesStyles`, `boundarySummary`, `residualPoints`, `coefficientSnapshot`, and `errorMetricSummary`.

- [ ] **Step 1: Write the failing Node tests**

Create `tests/visualization_models.test.js`:

```javascript
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const Models = require(
  "../.agents/skills/interactive-learning-experience-builder/assets/visualization-models.js",
);

const thresholdRecords = [
  { id: "c01", score: 0.92, actual: 1 },
  { id: "c02", score: 0.85, actual: 1 },
  { id: "c03", score: 0.78, actual: 0 },
  { id: "c04", score: 0.72, actual: 1 },
  { id: "c05", score: 0.66, actual: 1 },
  { id: "c06", score: 0.58, actual: 0 },
  { id: "c07", score: 0.49, actual: 1 },
  { id: "c08", score: 0.43, actual: 0 },
  { id: "c09", score: 0.35, actual: 1 },
  { id: "c10", score: 0.28, actual: 0 },
  { id: "c11", score: 0.18, actual: 0 },
  { id: "c12", score: 0.08, actual: 0 },
];

test("threshold summary computes counts, precision, and recall", () => {
  assert.deepEqual(Models.thresholdSummary(thresholdRecords, 0.5), {
    threshold: 0.5,
    tp: 4,
    fp: 2,
    tn: 4,
    fn: 2,
    precision: 2 / 3,
    recall: 2 / 3,
  });
});

test("threshold summary represents zero denominators as null", () => {
  const noPositivePredictions = [
    { id: "n1", score: 0.1, actual: 1 },
    { id: "n2", score: 0.2, actual: 0 },
  ];
  const noActualPositives = [
    { id: "n1", score: 0.9, actual: 0 },
    { id: "n2", score: 0.1, actual: 0 },
  ];

  assert.equal(Models.thresholdSummary(noPositivePredictions, 1).precision, null);
  assert.equal(Models.thresholdSummary(noActualPositives, 0.5).recall, null);
});

test("series styles depend on sorted series labels, not point order", () => {
  assert.deepEqual(Models.seriesStyles(["B", "A", "B"]), {
    A: { colorRole: "primary", shape: "circle", pattern: "solid" },
    B: { colorRole: "secondary", shape: "square", pattern: "hatched" },
  });
  assert.deepEqual(Models.seriesStyles(["A", "B"]), Models.seriesStyles(["B", "A"]));
});

test("boundary summary predicts the positive series above the line", () => {
  const points = [
    { id: "a1", x: 1, y: 1.2, series: "A" },
    { id: "a2", x: 2.2, y: 2.4, series: "A" },
    { id: "b1", x: 2.7, y: 3, series: "B" },
    { id: "b2", x: 4, y: 4.1, series: "B" },
  ];
  const result = Models.boundarySummary(
    points,
    { id: "balanced", slope: -1, intercept: 5.3 },
    "B",
  );

  assert.equal(result.correct, 4);
  assert.equal(result.incorrect, 0);
  assert.deepEqual(
    result.points.map(({ id, predictedSeries }) => ({ id, predictedSeries })),
    [
      { id: "a1", predictedSeries: "A" },
      { id: "a2", predictedSeries: "A" },
      { id: "b1", predictedSeries: "B" },
      { id: "b2", predictedSeries: "B" },
    ],
  );
});

test("residual and coefficient calculations preserve signed semantics", () => {
  assert.deepEqual(
    Models.residualPoints({
      id: "curvature",
      points: [
        { id: "r1", x: 1, observed: 5, predicted: 3 },
        { id: "r2", x: 2, observed: 4, predicted: 5 },
      ],
    }),
    [
      { id: "r1", x: 1, observed: 5, predicted: 3, fitted: 3, residual: 2 },
      { id: "r2", x: 2, observed: 4, predicted: 5, fitted: 5, residual: -1 },
    ],
  );

  const snapshot = Models.coefficientSnapshot(
    {
      penalties: [0, 1],
      series: [
        { feature: "Area", ridge: [3, 1.2], lasso: [3, 0] },
        { feature: "Age", ridge: [-1.2, -0.7], lasso: [-1.2, 0] },
      ],
    },
    1,
  );
  assert.equal(snapshot.penalty, 1);
  assert.deepEqual(snapshot.rows, [
    { feature: "Area", ridge: 1.2, lasso: 0, lassoIsZero: true },
    { feature: "Age", ridge: -0.7, lasso: 0, lassoIsZero: true },
  ]);
});

test("large adjustable errors affect RMSE more than MAE", () => {
  const small = Models.errorMetricSummary([-2, -1, 0, 1, 2], 0);
  const large = Models.errorMetricSummary([-2, -1, 0, 1, 2], 20);

  assert.deepEqual(small.errors, [-2, -1, 0, 1, 2, 0]);
  assert.equal(small.mae, 1);
  assert.equal(small.mse, 10 / 6);
  assert.equal(small.rmse, Math.sqrt(10 / 6));
  assert.ok(large.rmse - small.rmse > large.mae - small.mae);
});
```

- [ ] **Step 2: Run the Node test and verify RED**

Run:

```powershell
node --test tests/visualization_models.test.js
```

Expected: FAIL with `Cannot find module ... visualization-models.js`.

- [ ] **Step 3: Implement the pure visualization model**

Create `.agents/skills/interactive-learning-experience-builder/assets/visualization-models.js`:

```javascript
(function visualizationModelsFactory(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.LearningVisualizationModels = api;
})(typeof globalThis === "object" ? globalThis : this, function buildModels() {
  "use strict";

  function safeRatio(numerator, denominator) {
    return denominator === 0 ? null : numerator / denominator;
  }

  function thresholdSummary(records, threshold) {
    const counts = { tp: 0, fp: 0, tn: 0, fn: 0 };
    records.forEach((record) => {
      const predicted = Number(record.score) >= threshold ? 1 : 0;
      const actual = Number(record.actual);
      if (predicted === 1 && actual === 1) counts.tp += 1;
      else if (predicted === 1 && actual === 0) counts.fp += 1;
      else if (predicted === 0 && actual === 0) counts.tn += 1;
      else counts.fn += 1;
    });
    return {
      threshold,
      ...counts,
      precision: safeRatio(counts.tp, counts.tp + counts.fp),
      recall: safeRatio(counts.tp, counts.tp + counts.fn),
    };
  }

  function seriesStyles(seriesNames) {
    const names = [...new Set(seriesNames.map(String))].sort();
    const styles = [
      { colorRole: "primary", shape: "circle", pattern: "solid" },
      { colorRole: "secondary", shape: "square", pattern: "hatched" },
    ];
    return Object.fromEntries(names.map((name, index) => [name, styles[index]]));
  }

  function boundarySummary(points, boundary, positiveSeries) {
    const allSeries = [...new Set(points.map((point) => String(point.series)))].sort();
    const negativeSeries = allSeries.find((series) => series !== positiveSeries);
    const mapped = points.map((point) => {
      const boundaryY = Number(boundary.slope) * Number(point.x) + Number(boundary.intercept);
      const predictedSeries = Number(point.y) >= boundaryY ? positiveSeries : negativeSeries;
      return {
        ...point,
        boundaryY,
        predictedSeries,
        correct: predictedSeries === String(point.series),
      };
    });
    const correct = mapped.filter((point) => point.correct).length;
    return { points: mapped, correct, incorrect: mapped.length - correct };
  }

  function residualPoints(scenario) {
    return scenario.points.map((point) => ({
      ...point,
      fitted: Number(point.predicted),
      residual: Number(point.observed) - Number(point.predicted),
    }));
  }

  function coefficientSnapshot(data, index) {
    return {
      penalty: Number(data.penalties[index]),
      rows: data.series.map((series) => ({
        feature: String(series.feature),
        ridge: Number(series.ridge[index]),
        lasso: Number(series.lasso[index]),
        lassoIsZero: Number(series.lasso[index]) === 0,
      })),
    };
  }

  function errorMetricSummary(baseErrors, adjustableError) {
    const errors = [...baseErrors.map(Number), Number(adjustableError)];
    const mae = errors.reduce((sum, value) => sum + Math.abs(value), 0) / errors.length;
    const mse = errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length;
    return { errors, mae, mse, rmse: Math.sqrt(mse) };
  }

  return {
    boundarySummary,
    coefficientSnapshot,
    errorMetricSummary,
    residualPoints,
    seriesStyles,
    thresholdSummary,
  };
});
```

- [ ] **Step 4: Run the Node tests and verify GREEN**

Run:

```powershell
node --test tests/visualization_models.test.js tests/quiz_state_machine.test.js
```

Expected: all visualization-model and existing quiz-state tests pass.

- [ ] **Step 5: Commit the calculation layer**

```powershell
git add .agents/skills/interactive-learning-experience-builder/assets/visualization-models.js tests/visualization_models.test.js
git commit -m "feat: add semantic visualization models"
```

---

### Task 2: Semantic Payload Validation and Deterministic Embedding

**Files:**
- Modify: `.agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py`
- Modify: `tests/test_lecture_site_generator.py`

**Interfaces:**
- Consumes: the five schemas in the approved design and the `visualization-models.js` asset from Task 1.
- Produces: `validate_visualization(visualization, location) -> list[str]`, `VISUALIZATION_MODELS_MARKER`, and deterministic model-source embedding through `render_site`, `generate_site`, and `write_site`.

- [ ] **Step 1: Add valid semantic fixtures and failing schema tests**

Add a `SEMANTIC_VISUALIZATIONS` dictionary to
`tests/test_lecture_site_generator.py` with these exact minimal valid values:

```python
import copy

SEMANTIC_VISUALIZATIONS = {
    "binary-threshold": {
        "id": "threshold",
        "type": "binary-threshold",
        "title": "Threshold",
        "explanation": "Thresholds convert scores to decisions.",
        "data": [
            {"id": "p1", "score": 0.9, "actual": 1},
            {"id": "p2", "score": 0.6, "actual": 0},
            {"id": "p3", "score": 0.4, "actual": 1},
            {"id": "p4", "score": 0.1, "actual": 0},
        ],
        "controls": {"minimum": 0.1, "maximum": 0.9, "step": 0.1, "initial": 0.5},
        "labels": {"positive": "Positive", "negative": "Negative"},
        "fallback": "At 0.5 there is one TP, one FP, one TN, and one FN.",
    },
    "labeled-scatter": {
        "id": "boundary",
        "type": "labeled-scatter",
        "title": "Boundary",
        "explanation": "A line divides two illustrative classes.",
        "data": [
            {"id": "a1", "x": 1, "y": 1, "series": "A"},
            {"id": "a2", "x": 2, "y": 1.5, "series": "A"},
            {"id": "b1", "x": 3, "y": 3, "series": "B"},
            {"id": "b2", "x": 4, "y": 4, "series": "B"},
        ],
        "controls": {
            "boundaries": [
                {
                    "id": "balanced",
                    "label": "Balanced boundary",
                    "slope": -1,
                    "intercept": 5,
                }
            ],
            "initial": "balanced",
        },
        "labels": {
            "x_axis": "Feature 1",
            "y_axis": "Feature 2",
            "series": {"A": "Class A", "B": "Class B"},
            "positive_series": "B",
        },
        "fallback": "Class A occupies the lower-left and Class B the upper-right.",
    },
    "residual-diagnostics": {
        "id": "residuals",
        "type": "residual-diagnostics",
        "title": "Residuals",
        "explanation": "Residual patterns reveal missed structure.",
        "data": {
            "scenarios": [
                {
                    "id": "appropriate",
                    "label": "Appropriate fit",
                    "points": [
                        {
                            "id": f"r{index}",
                            "x": index,
                            "observed": 2 * index + 1,
                            "predicted": 2 * index + 0.9,
                        }
                        for index in range(1, 6)
                    ],
                }
            ]
        },
        "controls": {"initial": "appropriate"},
        "labels": {
            "x_axis": "Predictor",
            "target_axis": "Observed target",
            "residual_axis": "Residual",
        },
        "fallback": "Residuals stay close to zero without a systematic pattern.",
    },
    "coefficient-path": {
        "id": "coefficients",
        "type": "coefficient-path",
        "title": "Coefficient paths",
        "explanation": "Ridge shrinks while Lasso can reach zero.",
        "data": {
            "penalties": [0, 0.5, 1],
            "series": [
                {"feature": "Area", "ridge": [3, 2, 1], "lasso": [3, 1.5, 0]},
                {"feature": "Age", "ridge": [-1, -0.7, -0.4], "lasso": [-1, -0.4, 0]},
            ],
        },
        "controls": {"initial_index": 0},
        "fallback": "Ridge values shrink; both Lasso values reach zero at penalty 1.",
    },
    "error-metrics": {
        "id": "metrics",
        "type": "error-metrics",
        "title": "Metric sensitivity",
        "explanation": "RMSE reacts more strongly to a large error.",
        "data": {"base_errors": [-2, -1, 0, 1, 2], "adjustable_error": [0, 5, 10]},
        "controls": {"initial_index": 1},
        "labels": {"units": "target units"},
        "fallback": "Increasing one error from 0 to 10 raises RMSE faster than MAE.",
    },
}
```

Add tests:

```python
@pytest.mark.parametrize("visualization", SEMANTIC_VISUALIZATIONS.values())
def test_validate_payload_accepts_semantic_visualization(
    payload: dict[str, object], visualization: dict[str, object]
) -> None:
    payload["visualizations"] = [visualization]
    assert validate_payload(payload) == []


@pytest.mark.parametrize(
    ("visualization_type", "field_path"),
    [
        ("binary-threshold", ("data", 0, "actual")),
        ("labeled-scatter", ("data", 0, "series")),
        ("residual-diagnostics", ("data", "scenarios", 0, "points")),
        ("coefficient-path", ("data", "series", 0, "lasso")),
        ("error-metrics", ("data", "adjustable_error")),
    ],
)
def test_validate_payload_rejects_broken_semantic_schema(
    payload: dict[str, object],
    visualization_type: str,
    field_path: tuple[object, ...],
) -> None:
    visualization = copy.deepcopy(SEMANTIC_VISUALIZATIONS[visualization_type])
    target: object = visualization
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    del target[field_path[-1]]  # type: ignore[index]
    payload["visualizations"] = [visualization]

    errors = validate_payload(payload)

    assert any(visualization_type in error for error in errors)
```

Also add a render test requiring one `__VISUALIZATION_MODELS__` marker to be
replaced and requiring a missing model source to raise `ValueError`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
uv run pytest tests/test_lecture_site_generator.py -q
```

Expected: failures report unsupported semantic types and the missing
visualization-model marker contract.

- [ ] **Step 3: Implement schema dispatch**

In `generate_learning_experience.py`:

```python
VISUALIZATION_MODELS_MARKER = "__VISUALIZATION_MODELS__"
VISUALIZATION_TYPES = {
    "histogram",
    "boxplot",
    "scatter",
    "missingness",
    "binary-threshold",
    "labeled-scatter",
    "residual-diagnostics",
    "coefficient-path",
    "error-metrics",
}


def validate_visualization(
    visualization: dict[str, object],
    location: str,
) -> list[str]:
    visualization_type = visualization.get("type")
    validators = {
        "binary-threshold": _validate_binary_threshold,
        "labeled-scatter": _validate_labeled_scatter,
        "residual-diagnostics": _validate_residual_diagnostics,
        "coefficient-path": _validate_coefficient_path,
        "error-metrics": _validate_error_metrics,
    }
    if visualization_type in validators:
        return validators[visualization_type](visualization, location)
    errors: list[str] = []
    if not _has_valid_visualization_data(visualization_type, visualization.get("data")):
        errors.append(f"{location}.data does not match the {visualization_type} schema")
    if visualization_type == "histogram" and not _has_valid_histogram_bins(
        visualization.get("controls")
    ):
        errors.append(f"{location}.controls.bins must contain positive integers up to 50")
    return errors
```

Implement the five private validators with the exact field, uniqueness,
numeric range, array length, label, initial-value, and cross-reference rules
from the approved specification. Every returned message starts with
`{location}.<field>` and includes the visualization type, for example:

```python
errors.append(
    f"{location}.data does not match the binary-threshold schema: "
    "records need unique IDs, scores from 0 through 1, and actual values 0 or 1"
)
```

Replace the existing generic `_has_valid_visualization_data` call inside
`validate_payload` with:

```python
errors.extend(validate_visualization(visualization, location))
```

- [ ] **Step 4: Embed the visualization model deterministically**

Extend the render signatures:

```python
def render_site(
    template: str,
    payload: dict[str, object],
    quiz_state_machine: str | None = None,
    visualization_models: str | None = None,
) -> str:
```

Validate at most one marker, require the model source when the marker exists,
and replace it after the content and quiz replacements:

```python
model_marker_count = template.count(VISUALIZATION_MODELS_MARKER)
if model_marker_count > 1:
    raise ValueError(
        "template must contain no more than one "
        f"{VISUALIZATION_MODELS_MARKER} marker; found {model_marker_count}"
    )
if model_marker_count == 1 and visualization_models is None:
    raise ValueError("template requires the embedded visualization models")
if model_marker_count == 1:
    assert visualization_models is not None
    rendered = rendered.replace(VISUALIZATION_MODELS_MARKER, visualization_models)
```

Pass the new argument through `generate_site`. In `write_site`, load:

```python
visualization_models = None
if VISUALIZATION_MODELS_MARKER in template:
    visualization_models = template_path.with_name("visualization-models.js").read_text(
        encoding="utf-8"
    )
```

Then call:

```python
html = generate_site(payload, template, quiz_state_machine, visualization_models)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
uv run ruff format .agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py tests/test_lecture_site_generator.py
uv run pytest tests/test_lecture_site_generator.py -q
```

Expected: all generator tests pass.

- [ ] **Step 6: Commit the contract implementation**

```powershell
git add .agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py tests/test_lecture_site_generator.py
git commit -m "feat: validate semantic visualization payloads"
```

---

### Task 3: Accessible Semantic Renderers and Perceptible Palette

**Files:**
- Modify: `.agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html`
- Modify: `.agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py`
- Modify: `tests/test_lecture_site_generator.py`

**Interfaces:**
- Consumes: `LearningVisualizationModels` from Task 1 and validated payloads from Task 2.
- Produces: five interactive SVG renderers, `#palette-status`, graph-specific CSS variables, and validator-enforced runtime hooks.

- [ ] **Step 1: Add failing template and validator tests**

Extend the stable hook parameterization in `tests/test_lecture_site_generator.py`
with:

```python
[
    "__VISUALIZATION_MODELS__",
    'id="palette-status"',
    "--graph-primary: #6d28d9",
    "--graph-secondary: #0f766e",
    "--graph-primary: #0072b2",
    "--graph-secondary: #d55e00",
    "renderBinaryThreshold",
    "renderLabeledScatter",
    "renderResidualDiagnostics",
    "renderCoefficientPath",
    "renderErrorMetrics",
    "LearningVisualizationModels.thresholdSummary",
    "LearningVisualizationModels.seriesStyles",
]
```

Add:

```python
def test_palette_modes_use_visibly_different_graph_tokens() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "--graph-primary: #6d28d9" in template
    assert "--graph-secondary: #0f766e" in template
    assert "--graph-primary: #0072b2" in template
    assert "--graph-secondary: #d55e00" in template
    assert "Palette: standard" in template
    assert "Palette: color-blind-safe" in template


def test_validate_html_requires_palette_status(tmp_path: Path) -> None:
    path = tmp_path / "index.html"
    path.write_text(_valid_html().replace('id="palette-status"', ""), encoding="utf-8")
    assert any("palette-status" in error for error in validate_html(path))
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
uv run pytest tests/test_lecture_site_generator.py -q
```

Expected: failures identify missing renderer hooks, graph tokens, model marker,
and palette status validation.

- [ ] **Step 3: Add graph tokens, status, and model marker**

In `learning-experience-template.html`, add to `:root`:

```css
--graph-primary: #6d28d9;
--graph-primary-strong: #4c1d95;
--graph-secondary: #0f766e;
--graph-secondary-strong: #115e59;
```

Add to `:root[data-color-blind="true"]`:

```css
--graph-primary: #0072b2;
--graph-primary-strong: #00547f;
--graph-secondary: #d55e00;
--graph-secondary-strong: #9f4500;
```

Change `.mark-primary` and `.mark-secondary` to use graph variables, add
`.mark-series-primary`, `.mark-series-secondary`, `.boundary-line`,
`.zero-line`, `.metric-bar`, `.confusion-cell`, and `.coefficient-zero` styles,
and preserve diagonal hatching.

Place:

```html
<p id="palette-status" class="setting-status" aria-live="polite">
  Palette: standard
</p>
```

beside the color-blind checkbox, and place:

```html
<script>__VISUALIZATION_MODELS__</script>
```

before the template runtime script.

In `applySettings`, set:

```javascript
byId("palette-status").textContent = state.colorBlind
  ? "Palette: color-blind-safe"
  : "Palette: standard";
```

- [ ] **Step 4: Implement all five renderers**

Add these functions next to the existing renderers:

```javascript
function renderBinaryThreshold(visualization, target, summary, threshold) {
  const result = LearningVisualizationModels.thresholdSummary(
    visualization.data,
    threshold,
  );
  const formatMetric = (value) => value === null ? "not defined" : value.toFixed(2);
  target.innerHTML = `<div class="threshold-layout">
    <svg viewBox="0 0 520 250" role="img"
      aria-label="${escapeHtml(visualization.title)} confusion matrix">
      <text class="chart-label" x="240" y="24">Predicted negative</text>
      <text class="chart-label" x="385" y="24">Predicted positive</text>
      <text class="chart-label" x="12" y="95">Actual negative</text>
      <text class="chart-label" x="12" y="190">Actual positive</text>
      <g class="confusion-cell outcome-tn">
        <rect x="190" y="45" width="135" height="80"></rect>
        <text x="245" y="88">TN ${result.tn}</text>
      </g>
      <g class="confusion-cell outcome-fp">
        <rect x="335" y="45" width="135" height="80"></rect>
        <text x="390" y="88">FP ${result.fp}</text>
      </g>
      <g class="confusion-cell outcome-fn">
        <rect x="190" y="140" width="135" height="80"></rect>
        <text x="245" y="183">FN ${result.fn}</text>
      </g>
      <g class="confusion-cell outcome-tp">
        <rect x="335" y="140" width="135" height="80"></rect>
        <text x="390" y="183">TP ${result.tp}</text>
      </g>
    </svg>
  </div>`;
  summary.textContent =
    `Threshold ${threshold.toFixed(2)}. TP ${result.tp}, FP ${result.fp}, ` +
    `TN ${result.tn}, FN ${result.fn}. Precision ${formatMetric(result.precision)}; ` +
    `recall ${formatMetric(result.recall)}.`;
}
```

Implement `renderLabeledScatter` with stable styles from
`LearningVisualizationModels.seriesStyles`, a selected boundary from
`visualization.controls.boundaries`, direct series labels in every SVG
`<title>`, meaningful axis labels from the payload, and a dashed boundary.
Predict the positive series above the line through
`LearningVisualizationModels.boundarySummary`. Its summary is:

```javascript
summary.textContent =
  `${boundary.label}. ${result.correct} of ${result.points.length} illustrative ` +
  `points are on the expected side; ${result.incorrect} are not.`;
```

Implement `renderResidualDiagnostics` with two stacked SVG panels. Use
`LearningVisualizationModels.residualPoints` and payload labels. The top panel
shows observed points and predicted points connected by residual segments. The
bottom panel shows residual against fitted value with a dashed zero line. Its
summary identifies the selected scenario label and reports minimum and maximum
residuals without claiming a formal test.

Implement `renderCoefficientPath` with one polyline per feature and model,
selected-step markers, a penalty slider, exact-zero Lasso markers, and a text
table of the current Ridge and Lasso values from
`LearningVisualizationModels.coefficientSnapshot`.

Implement `renderErrorMetrics` with base-error marks, a distinct adjustable
error mark, and MAE/MSE/RMSE bars. Use
`LearningVisualizationModels.errorMetricSummary` and report metrics to two
decimal places with the payload units.

For each type, add a control branch to `visualizationControl`:

```javascript
if (visualization.type === "binary-threshold") {
  const controls = visualization.controls;
  return `<label>Decision threshold
    <input type="range" min="${controls.minimum}" max="${controls.maximum}"
      step="${controls.step}" value="${controls.initial}"
      data-viz-control="${escapeHtml(visualization.id)}">
  </label>`;
}
```

Use selects for labeled-scatter boundaries and residual scenarios, and range
inputs over array indexes for coefficient-path and error-metrics.

Extend `renderOneVisualization` with five explicit branches and keep the four
existing branches unchanged.

- [ ] **Step 5: Extend offline HTML validation**

In `validate_learning_experience.py`, require:

- one non-empty `#palette-status`;
- embedded `LearningVisualizationModels`;
- no unreplaced `__VISUALIZATION_MODELS__` marker;
- existing readable graph fallbacks.

Return direct errors:

```python
errors.append("missing live palette status")
errors.append("missing embedded visualization models")
errors.append("unreplaced visualization model marker")
```

- [ ] **Step 6: Format and verify the renderer contract**

Run:

```powershell
uv run ruff format .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py tests/test_lecture_site_generator.py
uv run pytest tests/test_lecture_site_generator.py -q
node --test tests/visualization_models.test.js tests/quiz_state_machine.test.js
```

Expected: all renderer, validator, and Node tests pass.

- [ ] **Step 7: Commit the renderer**

```powershell
git add .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py tests/test_lecture_site_generator.py
git commit -m "feat: render accessible semantic visualizations"
```

---

### Task 4: Classification Threshold and Boundary Experience

**Files:**
- Modify: `tests/test_classification_part_1_lecture_experience.py`
- Modify: `lecture_experiences/content/lecture_05_classification_part_1.json`
- Regenerate: `lecture_experiences/lecture_05_classification_part_1/index.html`

**Interfaces:**
- Consumes: `binary-threshold` and `labeled-scatter` contracts from Tasks 2–3.
- Produces: a grounded Classification companion with two semantically distinct interactions.

- [ ] **Step 1: Write failing Classification contract tests**

Require:

```python
visualizations = payload["visualizations"]
assert [visualization["type"] for visualization in visualizations] == [
    "binary-threshold",
    "labeled-scatter",
]

threshold = visualizations[0]
assert threshold["controls"]["initial"] == 0.5
assert {record["actual"] for record in threshold["data"]} == {0, 1}

boundary = visualizations[1]
assert {point["series"] for point in boundary["data"]} == {"A", "B"}
assert boundary["labels"]["positive_series"] == "B"
assert len(boundary["controls"]["boundaries"]) == 3

html = HTML_PATH.read_text(encoding="utf-8")
for hook in (
    "Decision threshold",
    "confusion matrix",
    "Precision",
    "recall",
    "Balanced boundary",
    "Conservative positive boundary",
    "Permissive positive boundary",
):
    assert hook in html
```

- [ ] **Step 2: Run the Classification test and verify RED**

Run:

```powershell
uv run pytest tests/test_classification_part_1_lecture_experience.py -q
```

Expected: FAIL because the payload still contains histogram, scatter, and
boxplot.

- [ ] **Step 3: Replace the Classification visualizations**

Use these threshold records:

```json
[
  {"id":"c01","score":0.92,"actual":1},
  {"id":"c02","score":0.85,"actual":1},
  {"id":"c03","score":0.78,"actual":0},
  {"id":"c04","score":0.72,"actual":1},
  {"id":"c05","score":0.66,"actual":1},
  {"id":"c06","score":0.58,"actual":0},
  {"id":"c07","score":0.49,"actual":1},
  {"id":"c08","score":0.43,"actual":0},
  {"id":"c09","score":0.35,"actual":1},
  {"id":"c10","score":0.28,"actual":0},
  {"id":"c11","score":0.18,"actual":0},
  {"id":"c12","score":0.08,"actual":0}
]
```

Use controls:

```json
{"minimum":0.1,"maximum":0.9,"step":0.05,"initial":0.5}
```

At 0.5, the fallback states `TP 4, FP 2, TN 4, FN 2; precision and recall are
both about 0.67`.

Use the existing twelve grounded class points but replace `class` with
`series`, add stable IDs, and use these boundaries:

```json
[
  {
    "id":"balanced",
    "label":"Balanced boundary",
    "slope":-1.0,
    "intercept":5.3
  },
  {
    "id":"conservative",
    "label":"Conservative positive boundary",
    "slope":-1.0,
    "intercept":6.0
  },
  {
    "id":"permissive",
    "label":"Permissive positive boundary",
    "slope":-1.0,
    "intercept":4.6
  }
]
```

Use meaningful axis labels `Feature 1` and `Feature 2`, display labels
`Class A` and `Class B`, and positive series `B`. Cite only the existing
Lecture 05 README and notes already named in the payload.

- [ ] **Step 4: Regenerate Classification deterministically**

Run:

```powershell
$slug = 'lecture_05_classification_part_1'
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py `
  --lecture-slug $slug `
  --content "lecture_experiences/content/$slug.json" `
  --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
  --output "lecture_experiences/$slug/index.html"

uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  "lecture_experiences/$slug/index.html"
```

Expected: `GENERATED` followed by `VALID`.

- [ ] **Step 5: Verify Classification GREEN**

Run:

```powershell
uv run pytest tests/test_classification_part_1_lecture_experience.py tests/test_lecture_site_generator.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Classification**

```powershell
git add tests/test_classification_part_1_lecture_experience.py lecture_experiences/content/lecture_05_classification_part_1.json lecture_experiences/lecture_05_classification_part_1/index.html
git commit -m "feat: add classification decision explorers"
```

---

### Task 5: Regression Diagnostics, Regularization, and Metrics

**Files:**
- Modify: `tests/test_regression_lecture_experience.py`
- Modify: `lecture_experiences/content/lecture_04_regression.json`
- Regenerate: `lecture_experiences/lecture_04_regression/index.html`

**Interfaces:**
- Consumes: `residual-diagnostics`, `coefficient-path`, and `error-metrics` contracts from Tasks 2–3.
- Produces: a grounded Regression companion with all three approved interactions.

- [ ] **Step 1: Write failing Regression contract tests**

Require:

```python
visualizations = payload["visualizations"]
assert [visualization["type"] for visualization in visualizations] == [
    "residual-diagnostics",
    "coefficient-path",
    "error-metrics",
]

residuals = visualizations[0]
assert [scenario["id"] for scenario in residuals["data"]["scenarios"]] == [
    "appropriate",
    "curvature",
    "funnel",
]

paths = visualizations[1]
assert paths["data"]["penalties"] == [0, 0.1, 0.5, 1, 2]
assert all(series["lasso"][-1] == 0 for series in paths["data"]["series"])

metrics = visualizations[2]
assert metrics["data"]["adjustable_error"] == [0, 5, 10, 20]

html = HTML_PATH.read_text(encoding="utf-8")
for hook in (
    "Appropriate linear fit",
    "Curved residual pattern",
    "Funnel-shaped variance",
    "Ridge",
    "Lasso",
    "MAE",
    "RMSE",
):
    assert hook in html
```

- [ ] **Step 2: Run the Regression test and verify RED**

Run:

```powershell
uv run pytest tests/test_regression_lecture_experience.py -q
```

Expected: FAIL because the payload still contains scatter, histogram, and
boxplot.

- [ ] **Step 3: Add the three residual scenarios**

For every scenario use `x` values 1 through 7 and predictions
`[3, 5, 7, 9, 11, 13, 15]`.

Use residual arrays:

```text
appropriate: [0.2, -0.3, 0.1, 0.0, -0.2, 0.3, -0.1]
curvature:   [2.0, 0.0, -1.0, -2.0, -1.0, 0.0, 2.0]
funnel:      [-0.2, 0.3, -0.6, 0.8, -1.4, 1.8, -2.4]
```

For each point set `observed = predicted + residual`, use IDs
`appropriate-1` through `funnel-7`, and use labels:

```json
{
  "x_axis":"Predictor",
  "target_axis":"Observed target",
  "residual_axis":"Residual (observed - predicted)"
}
```

The fallbacks state:

- appropriate: residuals remain close to zero without a systematic pattern;
- curvature: residuals are positive at both ends and negative in the middle;
- funnel: residual magnitude grows as fitted values increase.

- [ ] **Step 4: Add fixed Ridge and Lasso paths**

Use:

```json
{
  "penalties":[0,0.1,0.5,1,2],
  "series":[
    {
      "feature":"Area",
      "ridge":[3,2.7,2.2,1.7,1.1],
      "lasso":[3,2.6,1.8,0.8,0]
    },
    {
      "feature":"Rooms",
      "ridge":[1.8,1.6,1.2,0.9,0.5],
      "lasso":[1.8,1.4,0.7,0,0]
    },
    {
      "feature":"Age",
      "ridge":[-1.2,-1.1,-0.9,-0.7,-0.4],
      "lasso":[-1.2,-0.9,-0.3,0,0]
    }
  ]
}
```

Set `initial_index` to `0`. State explicitly that these are illustrative
precomputed paths, that Ridge shrinks without feature selection here, and that
Lasso zeros do not establish causal irrelevance.

- [ ] **Step 5: Add metric sensitivity data**

Use:

```json
{
  "base_errors":[-2,-1,0,1,2],
  "adjustable_error":[0,5,10,20]
}
```

Set `initial_index` to `1` and units to `target units`. The fallback compares
MAE and RMSE at adjustable errors 0 and 20 and states that squared errors make
RMSE rise more strongly.

- [ ] **Step 6: Regenerate Regression deterministically**

Run:

```powershell
$slug = 'lecture_04_regression'
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py `
  --lecture-slug $slug `
  --content "lecture_experiences/content/$slug.json" `
  --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
  --output "lecture_experiences/$slug/index.html"

uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  "lecture_experiences/$slug/index.html"
```

Expected: `GENERATED` followed by `VALID`.

- [ ] **Step 7: Verify Regression GREEN**

Run:

```powershell
uv run pytest tests/test_regression_lecture_experience.py tests/test_lecture_site_generator.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Regression**

```powershell
git add tests/test_regression_lecture_experience.py lecture_experiences/content/lecture_04_regression.json lecture_experiences/lecture_04_regression/index.html
git commit -m "feat: add regression diagnostic explorers"
```

---

### Task 6: Skill, Contract, and Course-Adapter Guidance

**Files:**
- Modify: `.agents/skills/interactive-learning-experience-builder/SKILL.md`
- Modify: `.agents/skills/interactive-learning-experience-builder/references/content-contract.md`
- Modify: `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md`
- Modify: `tests/test_interactive_learning_experience_builder_skill.py`
- Modify: `tests/test_interactive_learning_assistant_skill.py`

**Interfaces:**
- Consumes: proven renderer and payload behavior from Tasks 1–5.
- Produces: agent instructions that cause future experiences to select semantic visualizations, preserve semantic fields, and verify perceptible palette behavior.

- [ ] **Step 1: Add failing skill-documentation tests**

Require these exact portable-core phrases:

```python
for phrase in (
    "topic-relevant interpretation",
    "Do not discard semantic payload fields",
    "meaningful axis, series, scenario, and control labels",
    "visibly different graph marks",
    "binary-threshold",
    "labeled-scatter",
    "residual-diagnostics",
    "coefficient-path",
    "error-metrics",
):
    assert phrase in skill_text or phrase in contract_text
```

Require these exact adapter phrases:

```python
for phrase in (
    "named lecture objective",
    "repeated generic chart set",
    "semantic visualization type",
    "exercise every visualization control",
    "both palette modes",
):
    assert phrase in adapter_text
```

- [ ] **Step 2: Run skill tests and verify RED**

Run:

```powershell
uv run pytest tests/test_interactive_learning_experience_builder_skill.py tests/test_interactive_learning_assistant_skill.py -q
```

Expected: failures identify missing semantic-visualization and palette
guidance.

- [ ] **Step 3: Update the portable skill**

Add this concise rule block under non-negotiable experience rules:

```markdown
- Give at least one visualization a control that changes a topic-relevant
  interpretation, not only its presentation.
- Do not discard semantic payload fields such as class, series, scenario,
  outcome, coefficient, or unit labels.
- Use meaningful axis, series, scenario, and control labels.
- Verify that standard and color-blind-safe modes produce visibly different
  graph marks while preserving non-color cues.
```

Add a type-selection table to the content contract listing all nine supported
types, their learning purpose, required data shape, and control shape. Copy the
five exact schema rules from the approved specification without weakening
their validation boundaries.

- [ ] **Step 4: Update the ML-course adapter**

Add:

```markdown
For every visualization, name the lecture objective it teaches. Do not repeat
a generic chart set across lectures when a supported semantic visualization
type better represents the objective. Exercise every visualization control
and both palette modes before publishing.
```

Retain all existing public-source, exclusion, output, and publishing rules.

- [ ] **Step 5: Verify skill tests GREEN**

Run:

```powershell
uv run pytest tests/test_interactive_learning_experience_builder_skill.py tests/test_interactive_learning_assistant_skill.py -q
```

Expected: all skill contract tests pass.

- [ ] **Step 6: Commit skill guidance**

```powershell
git add .agents/skills/interactive-learning-experience-builder/SKILL.md .agents/skills/interactive-learning-experience-builder/references/content-contract.md .agents/skills/ml-course-interactive-learning-assistant/SKILL.md tests/test_interactive_learning_experience_builder_skill.py tests/test_interactive_learning_assistant_skill.py
git commit -m "docs: teach semantic visualization authoring"
```

---

### Task 7: Regenerate EDA and Integrate CI Coverage

**Files:**
- Modify: `tests/test_eda_lecture_experience.py`
- Regenerate: `lecture_experiences/lecture_01_eda/index.html`
- Modify: `.github/workflows/build-textbook-preview.yml`
- Modify: `.github/workflows/validate-okf.yml`

**Interfaces:**
- Consumes: shared template and visualization model from Tasks 1–3.
- Produces: regenerated EDA with the new palette behavior and hosted coverage for all new Python and Node tests.

- [ ] **Step 1: Add failing EDA shared-runtime assertions**

Add:

```python
html = HTML_PATH.read_text(encoding="utf-8")
for hook in (
    "LearningVisualizationModels",
    'id="palette-status"',
    "Palette: color-blind-safe",
    "--graph-primary:#6d28d9",
    "--graph-primary:#0072b2",
):
    assert hook.replace(" ", "") in html.replace(" ", "")
assert [visualization["type"] for visualization in payload["visualizations"]] == [
    "histogram",
    "boxplot",
    "scatter",
    "missingness",
]
```

Normalize only spaces for compact generated CSS assertions; do not loosen the
visualization order or type assertions.

- [ ] **Step 2: Run EDA test and verify RED**

Run:

```powershell
uv run pytest tests/test_eda_lecture_experience.py -q
```

Expected: failure because the committed EDA artifact predates the model and
palette update.

- [ ] **Step 3: Regenerate and validate EDA**

Run:

```powershell
$slug = 'lecture_01_eda'
uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py `
  --lecture-slug $slug `
  --content "lecture_experiences/content/$slug.json" `
  --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
  --output "lecture_experiences/$slug/index.html"

uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py `
  "lecture_experiences/$slug/index.html"
```

Expected: `GENERATED` followed by `VALID`.

- [ ] **Step 4: Expand both GitHub workflows**

Add these paths to push/pull-request triggers where applicable:

```yaml
- "tests/test_regression_lecture_experience.py"
- "tests/test_classification_part_1_lecture_experience.py"
- "tests/visualization_models.test.js"
```

Add both Python experience tests to the hosted pytest commands and add:

```yaml
- name: Run browser behavior models
  run: node --test tests/quiz_state_machine.test.js tests/visualization_models.test.js
```

Keep Node 22 and all existing gates.

- [ ] **Step 5: Verify EDA and workflow-facing tests GREEN**

Run:

```powershell
uv run pytest tests/test_eda_lecture_experience.py tests/test_regression_lecture_experience.py tests/test_classification_part_1_lecture_experience.py tests/test_textbook_preview.py -q
node --test tests/quiz_state_machine.test.js tests/visualization_models.test.js
```

Expected: all tests pass.

- [ ] **Step 6: Commit EDA and CI coverage**

```powershell
git add tests/test_eda_lecture_experience.py lecture_experiences/lecture_01_eda/index.html .github/workflows/build-textbook-preview.yml .github/workflows/validate-okf.yml
git commit -m "test: cover semantic companions in CI"
```

---

### Task 8: Repository, Student, Architecture, and Submission Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/learning-companions-architecture.md`
- Modify: `docs/interactive-lecture-learning-assistant.md`
- Modify: `docs/student-learning-companion-quickstart.md`
- Modify: `docs/build-week-integration-evidence.md`
- Modify: `docs/build-week-submission-preparation.md`
- Modify: `tests/test_interactive_learning_assistant_docs.py`

**Interfaces:**
- Consumes: verified behavior and final terminology from Tasks 1–7.
- Produces: one consistent explanation for students, agents, maintainers, judges, screenshots, and demo narration.

- [ ] **Step 1: Write failing documentation assertions**

Require:

```python
required_by_document = {
    "README.md": (
        "threshold and confusion matrix",
        "decision boundary",
        "residual diagnostics",
        "Ridge and Lasso",
        "MAE and RMSE",
    ),
    "docs/learning-companions-architecture.md": (
        "visualization-models.js",
        "semantic visualization payload",
        "pure visualization model",
    ),
    "docs/interactive-lecture-learning-assistant.md": (
        "topic-relevant interpretation",
        "both palette modes",
        "exercise every visualization control",
    ),
    "docs/student-learning-companion-quickstart.md": (
        "topic-specific interaction",
        "trusted source",
        "color-blind-safe",
    ),
    "docs/build-week-integration-evidence.md": (
        "ignored class semantics",
        "perceptually weak",
        "shared portable core",
    ),
    "docs/build-week-submission-preparation.md": (
        "feedback",
        "threshold",
        "regularization",
        "metric sensitivity",
        "palette",
    ),
}
for path, phrases in required_by_document.items():
    text = Path(path).read_text(encoding="utf-8")
    for phrase in phrases:
        assert phrase.lower() in text.lower()
```

- [ ] **Step 2: Run documentation tests and verify RED**

Run:

```powershell
uv run pytest tests/test_interactive_learning_assistant_docs.py -q
```

Expected: failures identify missing semantic-interaction and feedback-loop
language.

- [ ] **Step 3: Update repository and student entry points**

In `README.md`, replace the generic feature sentence with a short per-review
summary:

```markdown
- **EDA:** binning, IQR fences, association, and missingness.
- **Regression:** residual patterns, Ridge/Lasso shrinkage, and MAE/RMSE
  sensitivity.
- **Classification:** threshold-dependent confusion outcomes and class-aware
  decision boundaries.
```

In `docs/student-learning-companion-quickstart.md`, update the generic prompt
template to request one control that changes a topic-specific interpretation
and to require testing the color-blind-safe setting.

In `AGENTS.md`, add the visualization model, Node test, semantic payload, and
browser palette checks to the standalone review navigation and verification
sections.

- [ ] **Step 4: Update architecture and operational guidance**

Document this boundary in
`docs/learning-companions-architecture.md`:

```text
semantic payload
→ Python schema validation
→ embedded visualization-models.js calculations
→ template-owned SVG and DOM
→ live summary and static fallback
```

In `docs/interactive-lecture-learning-assistant.md`, require the author to name
the learning objective, exercise every control, inspect both palette modes,
and verify that semantic labels survive generation.

- [ ] **Step 5: Update integration evidence and submission preparation**

In `docs/build-week-integration-evidence.md`, add a dated iteration section
with three observed issues:

1. Regression and Classification repeated the same chart grammar.
2. Classification class labels were present in JSON but ignored by rendering.
3. The palette toggle changed computed colors but the two palettes were
   perceptually too similar.

Record the root-cause fix at the portable core and the exact automated,
browser, Actions, and Pages evidence.

In `docs/build-week-submission-preparation.md`, revise:

- the project-description field to mention semantic interactive companions;
- the judge test flow to exercise threshold, boundary, residual,
  regularization, metrics, and palette controls;
- the screenshot plan to capture Classification threshold/boundary and
  Regression regularization/metric sensitivity;
- the demo narration to show the generic core, repository adapter, source
  knowledge, feedback discovery, upstream fix, and regenerated outputs;
- the feedback section to frame this as learner-facing visual review followed
  by systematic improvement, not as a hidden defect.

- [ ] **Step 6: Verify documentation GREEN**

Run:

```powershell
uv run ruff format tests/test_interactive_learning_assistant_docs.py
uv run pytest tests/test_interactive_learning_assistant_docs.py -q
```

Expected: all documentation tests pass.

- [ ] **Step 7: Commit documentation**

```powershell
git add README.md AGENTS.md docs/learning-companions-architecture.md docs/interactive-lecture-learning-assistant.md docs/student-learning-companion-quickstart.md docs/build-week-integration-evidence.md docs/build-week-submission-preparation.md tests/test_interactive_learning_assistant_docs.py
git commit -m "docs: explain semantic learning companions"
```

---

### Task 9: Determinism, Full Verification, Browser Acceptance, and Publication

**Files:**
- Verify all files changed in Tasks 1–8.
- Generated preview: `site/_build/` (ignored, not committed).

**Interfaces:**
- Consumes: the complete semantic visualization implementation.
- Produces: verified commits, clean working tree, successful public `main`,
  successful GitHub Actions, and working live Pages routes.

- [ ] **Step 1: Prove deterministic regeneration**

For each slug:

```powershell
$slugs = @(
  'lecture_01_eda',
  'lecture_04_regression',
  'lecture_05_classification_part_1'
)
foreach ($slug in $slugs) {
  $temporary = Join-Path $env:TEMP "$slug-index.html"
  uv run python .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py `
    --lecture-slug $slug `
    --content "lecture_experiences/content/$slug.json" `
    --template .agents/skills/interactive-learning-experience-builder/assets/learning-experience-template.html `
    --output $temporary
  $expected = (Get-FileHash "lecture_experiences/$slug/index.html" -Algorithm SHA256).Hash
  $actual = (Get-FileHash $temporary -Algorithm SHA256).Hash
  if ($expected -ne $actual) { throw "Non-deterministic output for $slug" }
}
```

Expected: no exception; all hashes match.

- [ ] **Step 2: Run every standalone validator**

```powershell
uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py lecture_experiences/lecture_01_eda/index.html
uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py lecture_experiences/lecture_04_regression/index.html
uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py lecture_experiences/lecture_05_classification_part_1/index.html
```

Expected: three `VALID` results.

- [ ] **Step 3: Run complete automated verification**

```powershell
uv run ruff format --check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py .agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py tests/test_okf_validation.py tests/test_textbook_preview.py tests/test_textbook_contribution_skill.py tests/test_interactive_learning_assistant_docs.py tests/test_interactive_learning_assistant_skill.py tests/test_interactive_learning_experience_builder_skill.py tests/test_learning_experience_portability.py tests/test_lecture_site_generator.py tests/test_eda_lecture_experience.py tests/test_regression_lecture_experience.py tests/test_classification_part_1_lecture_experience.py
uv run ruff check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py .agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py .agents/skills/ml-course-interactive-learning-assistant/scripts/generate_course_learning_experience.py tests/test_okf_validation.py tests/test_textbook_preview.py tests/test_textbook_contribution_skill.py tests/test_interactive_learning_assistant_docs.py tests/test_interactive_learning_assistant_skill.py tests/test_interactive_learning_experience_builder_skill.py tests/test_learning_experience_portability.py tests/test_lecture_site_generator.py tests/test_eda_lecture_experience.py tests/test_regression_lecture_experience.py tests/test_classification_part_1_lecture_experience.py
node --test tests/quiz_state_machine.test.js tests/visualization_models.test.js
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
```

Expected:

- Ruff format and lint pass;
- all Node tests pass;
- pytest has zero failures;
- OKF has zero errors and zero warnings;
- the textbook builds at `site/_build`.

- [ ] **Step 4: Exercise local browser acceptance**

Open each committed `index.html` through `file://` at desktop and 390-pixel
mobile width.

Classification:

- move threshold to 0.30, 0.50, and 0.80 and verify counts and metrics;
- switch all three boundaries and verify class shapes stay tied to A/B;
- toggle both palette modes and verify purple/teal versus blue/vermillion;
- use keyboard-only operation and verify visible focus.

Regression:

- switch appropriate, curvature, and funnel scenarios;
- move through every penalty step and verify Ridge shrinkage and Lasso zeros;
- move the adjustable error through 0, 5, 10, and 20 and verify RMSE reacts
  more strongly;
- toggle both palette modes and use keyboard-only operation.

EDA:

- exercise bins, fence multiplier, trend line, and missingness sort;
- verify both palettes visibly change marks while current EDA semantics remain.

All pages:

- verify live summaries after each control;
- verify static fallbacks, focus mode, reduced motion, storage-disabled
  behavior, no horizontal page overflow, and no network requests.

- [ ] **Step 5: Inspect the final diff and commit verification-only fixes**

Run:

```powershell
git diff --check
git status --short
git log --oneline upstream/main..HEAD
```

Expected: no uncommitted source changes and only the intended semantic
visualization commits ahead of public `main`. If formatting changed during
verification, commit only those mechanical changes as:

```powershell
git add --update
git commit -m "style: format semantic visualization checks"
```

- [ ] **Step 6: Publish the verified branch to the public repository**

Confirm the fast-forward:

```powershell
git fetch upstream main
git merge-base --is-ancestor upstream/main HEAD
```

Then push without force:

```powershell
git push upstream HEAD:main
```

Expected: public `main` advances to the verified HEAD. Do not push or merge
into the private teacher repository.

- [ ] **Step 7: Wait for GitHub Actions**

Use:

```powershell
$sha = git rev-parse HEAD
gh run list --repo DerAndr/machine_learning_course_basics --commit $sha `
  --json databaseId,name,status,conclusion,url
```

Watch both runs with `gh run watch <run-id> --exit-status`. Expected:

- `Validate OKF`: success;
- `Build Textbook Preview`: success, including Pages deployment.

- [ ] **Step 8: Verify live Pages**

Request these routes and require HTTP 200 plus the named marker:

```text
https://derandr.github.io/machine_learning_course_basics/
  marker: Fast interactive reviews
https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/
  marker: Palette
https://derandr.github.io/machine_learning_course_basics/demos/lecture_04_regression/
  marker: Ridge
https://derandr.github.io/machine_learning_course_basics/demos/lecture_05_classification_part_1/
  marker: Decision threshold
```

Expected: all four routes return 200 and contain the expected deployed
content.
