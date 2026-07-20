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
