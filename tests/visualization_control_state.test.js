"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.resolve(__dirname, "..");

function attribute(attributes, name) {
  const match = attributes.match(new RegExp(`\\b${name}="([^"]*)"`));
  return match ? match[1] : "";
}

class FakeControl {
  constructor(tagName, attributes, body = "") {
    this.tagName = tagName.toUpperCase();
    this.type = attribute(attributes, "type");
    this.value = attribute(attributes, "value");
    this.checked = /\bchecked(?:\s|>|$)/.test(attributes);
    this.listeners = new Map();

    if (tagName === "select") {
      const options = [...body.matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)];
      const selected = options.find((option) => /\bselected(?:\s|>|$)/.test(option[1]));
      const initial = selected || options[0];
      this.value = initial ? attribute(initial[1], "value") : "";
    }
  }

  addEventListener(eventType, listener) {
    this.listeners.set(eventType, listener);
  }

  dispatch(eventType) {
    const listener = this.listeners.get(eventType);
    assert.ok(listener, `expected ${eventType} listener`);
    listener({ target: this });
  }

  matches(selector) {
    return selector === 'input[type="range"]' && this.tagName === "INPUT"
      && this.type === "range";
  }
}

function generatedRuntime(lectureSlug) {
  const html = fs.readFileSync(
    path.join(ROOT, "lecture_experiences", lectureSlug, "index.html"),
    "utf8",
  );
  const contentPrefix = "<script>const CONTENT = ";
  const contentStart = html.indexOf(contentPrefix);
  const contentEnd = html.indexOf(";</script>", contentStart);
  assert.notEqual(contentStart, -1, "generated HTML must embed CONTENT");
  assert.notEqual(contentEnd, -1, "generated HTML must close embedded CONTENT");
  const CONTENT = JSON.parse(
    html.slice(contentStart + contentPrefix.length, contentEnd),
  );

  const runtimeStart = html.indexOf("    function readVisualizationControlValue(");
  const runtimeEnd = html.indexOf("    function currentQuestions()", runtimeStart);
  assert.notEqual(runtimeStart, -1, "generated HTML must include visualization control state");
  assert.notEqual(runtimeEnd, -1, "generated HTML must include visualization lifecycle");
  const runtimeSource = html.slice(runtimeStart, runtimeEnd);

  const elements = new Map();
  const controls = new Map();
  const renderCalls = new Map();
  const visualizationList = {
    value: "",
    set innerHTML(value) {
      this.value = value;
      controls.clear();

      for (const match of value.matchAll(
        /<select\b([^>]*data-viz-control="([^"]+)"[^>]*)>([\s\S]*?)<\/select>/g,
      )) {
        controls.set(match[2], new FakeControl("select", match[1], match[3]));
      }
      for (const match of value.matchAll(
        /<input\b([^>]*data-viz-control="([^"]+)"[^>]*)>/g,
      )) {
        controls.set(match[2], new FakeControl("input", match[1]));
      }
    },
    get innerHTML() {
      return this.value;
    },
  };
  elements.set("visualization-list", visualizationList);
  elements.set("previous-visualization", { disabled: false });
  elements.set("next-visualization", { disabled: false });

  const state = {
    colorBlind: CONTENT.defaults.color_blind,
    focus: CONTENT.defaults.focus_mode,
    visualizationIndex: 0,
    visualizationControlValues: {},
  };
  const byId = (id) => {
    if (!elements.has(id)) {
      elements.set(id, { innerHTML: "", textContent: "", disabled: false });
    }
    return elements.get(id);
  };
  const document = {
    querySelector(selector) {
      const match = selector.match(/^\[data-viz-control="([^"]+)"\]$/);
      return match ? controls.get(match[1]) : null;
    },
  };
  const record = (type) => (visualization, _target, _summary, value) => {
    renderCalls.set(visualization.id, { type, value });
  };
  const context = {
    CONTENT,
    CSS: { escape: (value) => value },
    state,
    document,
    byId,
    escapeHtml: (value) => String(value),
    renderHistogram: record("histogram"),
    renderBoxplot: record("boxplot"),
    renderScatter: record("scatter"),
    renderMissingness: record("missingness"),
    renderBinaryThreshold: record("binary-threshold"),
    renderLabeledScatter: record("labeled-scatter"),
    renderResidualDiagnostics: record("residual-diagnostics"),
    renderCoefficientPath: record("coefficient-path"),
    renderErrorMetrics: record("error-metrics"),
  };
  vm.runInNewContext(`${runtimeSource}\nthis.renderVisualizations = renderVisualizations;`, context);

  return {
    CONTENT,
    state,
    control(id) {
      const control = controls.get(id);
      assert.ok(control, `expected control for ${id}`);
      return control;
    },
    render() {
      context.renderVisualizations();
    },
    change(id, value, eventType) {
      const control = this.control(id);
      if (typeof value === "boolean") {
        control.checked = value;
      } else {
        control.value = String(value);
      }
      control.dispatch(eventType);
    },
    lastRender(id) {
      return renderCalls.get(id);
    },
  };
}

test("threshold range survives palette changes and previous/next graph navigation", () => {
  const app = generatedRuntime("lecture_05_classification_part_1");
  const thresholdId = "cls-threshold-confusion";

  app.render();
  app.change(thresholdId, "0.8", "input");
  assert.equal(app.lastRender(thresholdId).value, 0.8);

  app.state.colorBlind = !app.state.colorBlind;
  app.render();
  assert.equal(app.control(thresholdId).value, "0.8");
  assert.equal(app.lastRender(thresholdId).value, 0.8);

  app.state.visualizationIndex = 1;
  app.render();
  app.state.visualizationIndex = 0;
  app.render();
  assert.equal(app.control(thresholdId).value, "0.8");
});

test("boundary and residual selects survive focus and palette rerenders independently", () => {
  const classification = generatedRuntime("lecture_05_classification_part_1");
  classification.render();
  classification.change("cls-decision-boundary", "conservative", "change");
  classification.state.focus = !classification.state.focus;
  classification.render();
  classification.state.colorBlind = !classification.state.colorBlind;
  classification.render();
  assert.equal(classification.control("cls-decision-boundary").value, "conservative");
  assert.equal(classification.control("cls-threshold-confusion").value, "0.5");

  const regression = generatedRuntime("lecture_04_regression");
  const residual = regression.CONTENT.visualizations.find(
    (visualization) => visualization.type === "residual-diagnostics",
  );
  assert.ok(residual);
  const alternate = residual.data.scenarios.find(
    (scenario) => scenario.id !== residual.controls.initial,
  );
  assert.ok(alternate);
  regression.render();
  regression.change(residual.id, alternate.id, "change");
  regression.state.focus = !regression.state.focus;
  regression.render();
  regression.state.colorBlind = !regression.state.colorBlind;
  regression.render();
  assert.equal(regression.control(residual.id).value, alternate.id);
});

test("legacy select and checkbox controls survive rerender and navigation", () => {
  const app = generatedRuntime("lecture_01_eda");
  const histogram = app.CONTENT.visualizations.find(
    (visualization) => visualization.type === "histogram",
  );
  const scatter = app.CONTENT.visualizations.find(
    (visualization) => visualization.type === "scatter",
  );
  assert.ok(histogram);
  assert.ok(scatter);
  const alternateBins = histogram.controls.bins.at(-1);

  app.render();
  app.change(histogram.id, alternateBins, "change");
  app.change(scatter.id, true, "change");
  app.state.visualizationIndex = 2;
  app.render();
  app.state.visualizationIndex = 0;
  app.render();

  assert.equal(app.control(histogram.id).value, String(alternateBins));
  assert.equal(app.control(scatter.id).checked, true);
  assert.equal(app.lastRender(scatter.id).value, true);
});
