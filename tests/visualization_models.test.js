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
