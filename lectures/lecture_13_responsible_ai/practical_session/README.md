# Responsible AI Practical Session

This directory contains a 90-minute classroom practical for Lecture 13.

## Files

- `responsible_ai_practical_student_90min.ipynb`
- `responsible_ai_practical_student_90min.py`
- `README.md`

## Format

- The student notebook contains targeted TODO placeholders in fairness assessment, mitigation, comparison, and model-card cells.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- Helper functions are provided; students focus on applying them:
  - `load_adult_dataset` — fetch and prepare the UCI Adult Census dataset.
  - `build_metric_frame` — create a `MetricFrame` with 10 fairness-relevant metrics.
  - `scalar_fairness_summary` — print 8 scalar disparity metrics in one call.
  - `plot_fairness_dashboard` — 2×3 panel showing selection rate, TPR, FPR, FNR, accuracy, mean prediction by group.
  - `plot_metric_comparison` — side-by-side bar charts comparing metrics before and after mitigation.
  - `plot_intersectional_heatmap` — seaborn heatmap for any metric across sex × race subgroups.
- A **Fairness Metric Glossary** table is included in the notebook for quick reference.
- Fairlearn's `plot_model_comparison` scatter plot is used to visualize the performance-vs-fairness trade-off.

## Teaching Intent

- Give students hands-on experience with Fairlearn's `MetricFrame` and scalar disparity metrics.
- Compare multiple fairness definitions: demographic parity, equalized odds, equal opportunity.
- Understand the four-fifths rule and how to read fairness dashboards.
- Contrast post-processing (`ThresholdOptimizer`) and in-processing (`ExponentiatedGradient`) mitigation strategies.
- Analyze fairness across multiple sensitive features (sex and race), including an intersectional view.
- Introduce lightweight Model Card documentation as a responsible-AI practice.

## Scope Note

The lecture also covers topics not included in this 90-minute practical: privacy (differential privacy, federated learning), adversarial robustness, interpretability (SHAP, LIME — covered in Lecture 11), causal ML, conformal prediction, and pre-processing mitigation (data rebalancing). These are mentioned in the debrief for student awareness.
