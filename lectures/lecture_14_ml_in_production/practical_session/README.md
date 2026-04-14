# ML in Production Practical Session

This directory contains a 90-minute classroom practical for Lecture 14.

## Files

- `ml_in_production_practical_student_90min.ipynb` — student notebook with TODO placeholders
- `ml_in_production_practical_student_90min.py` — auto-generated companion script
- `README.md` — this file

## Format

- Student notebook with targeted TODO placeholders (9 exercises)
- Python companion script mirrors notebook structure
- Helper functions provided (load_adult_data, simulate_covariate_shift, simulate_label_shift, simulate_concept_drift, simulate_gradual_drift, plot_distribution_comparison, plot_monitoring_dashboard)
- Production Concepts Reference table included
- MLOps Tool Landscape comparison tables

## Teaching Intent

- Hands-on sklearn Pipeline construction (ColumnTransformer + Pipeline)
- Model serialization with metadata (joblib + JSON)
- Experiment tracking with MLflow (log params, metrics, artifacts)
- MLOps tool landscape overview (MLflow, W&B, ClearML, BentoML, Evidently, DVC, Airflow)
- Data validation gates (schema, nulls, ranges)
- Three drift types: covariate, label, concept — simulate, detect, compare
- Drift detection with Evidently (DataDriftPreset) and manual KS tests
- Monitoring dashboard over simulated time windows
- Canary release deployment strategy with rollback logic
- Production readiness checklist

## Scope Note

Topics covered conceptually but not as hands-on exercises: W&B and ClearML (require API keys), BentoML serving (requires Docker), full CI/CD, feature stores, A/B testing with real traffic, infrastructure autoscaling.
