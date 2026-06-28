function metricValue(numerator, denominator) {
  return denominator === 0 ? 0 : numerator / denominator;
}

function formatMetric(value) {
  return value.toFixed(2);
}

function computeMetrics(examples, threshold) {
  const counts = { tp: 0, fp: 0, tn: 0, fn: 0 };
  const rows = examples.map((example) => {
    const predicted = example.score >= threshold;
    const actual = Boolean(example.label);
    if (predicted && actual) counts.tp += 1;
    if (predicted && !actual) counts.fp += 1;
    if (!predicted && actual) counts.fn += 1;
    if (!predicted && !actual) counts.tn += 1;
    return { ...example, predicted };
  });
  const precision = metricValue(counts.tp, counts.tp + counts.fp);
  const recall = metricValue(counts.tp, counts.tp + counts.fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { counts, precision, recall, f1, rows };
}

function renderMetrics(container, metrics) {
  const cards = [
    ["TP", metrics.counts.tp],
    ["FP", metrics.counts.fp],
    ["TN", metrics.counts.tn],
    ["FN", metrics.counts.fn],
    ["Precision", formatMetric(metrics.precision)],
    ["Recall", formatMetric(metrics.recall)],
    ["F1", formatMetric(metrics.f1)],
  ];
  container.innerHTML = cards
    .map(([label, value]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderRows(container, rows) {
  container.innerHTML = `
    <table>
      <thead>
        <tr><th>Example</th><th>Score</th><th>Actual</th><th>Predicted</th></tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                <td>${row.id}</td>
                <td>${row.score.toFixed(2)}</td>
                <td>${row.label ? "positive" : "negative"}</td>
                <td>${row.predicted ? "positive" : "negative"}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

async function initThresholdLab(lab) {
  const response = await fetch(lab.dataset.url);
  const data = await response.json();
  const slider = lab.querySelector("#threshold-slider");
  const value = lab.querySelector("#threshold-value");
  const metricGrid = lab.querySelector(".metric-grid");
  const table = lab.querySelector(".example-table");

  function update() {
    const threshold = Number(slider.value);
    value.textContent = threshold.toFixed(2);
    const metrics = computeMetrics(data.examples, threshold);
    renderMetrics(metricGrid, metrics);
    renderRows(table, metrics.rows);
  }

  slider.addEventListener("input", update);
  update();
}

document.querySelectorAll("[data-threshold-lab]").forEach((lab) => {
  initThresholdLab(lab).catch((error) => {
    lab.insertAdjacentHTML(
      "beforeend",
      `<p role="alert">The interactive lab could not load. Use the static fallback table below.</p>`,
    );
    console.error(error);
  });
});
