"""ML in Production — Practical Session (90 min).

Auto-generated companion script.  Run the .ipynb for the full
interactive experience; this file mirrors the notebook structure
for quick reference and version-control diffs.
"""

# # ML in Production — Practical Session (90 min)
#
# **Lecture 14 — Machine Learning in Production**
#
# Welcome! In this session you play the role of an **ML engineer**.
# Your team trained a model in a Jupyter notebook. Your task: **make it production-ready**.
#
# ### Learning Objectives
#
# After completing this session you will be able to:
#
# 1. Build a **reproducible sklearn Pipeline** that packages preprocessing and model together
# 2. **Serialize models with metadata** for versioning and auditing
# 3. **Track experiments with MLflow** — log parameters, metrics, and artifacts
# 4. Navigate the **MLOps tool landscape** — W&B, ClearML, BentoML, and more
# 5. Implement **data validation gates** that catch schema and quality issues before inference
# 6. **Detect data drift** using statistical tests and the Evidently library
# 7. Simulate a **monitoring dashboard** that tracks model health over time
# 8. Practice **deployment strategy logic** (canary release with rollback)
# 9. Complete a **production readiness checklist**
#
# ### Format
#
# - 🟢 provided cells — run and read
# - ✏️ **TODO** cells — your turn to write code or answer questions
# - 💬 *Interpretation Guide* sections — group discussion prompts

# ## Setup
#
# **Local environment** (recommended):
#
# ```bash
# uv sync --group ml_in_production
# ```
#
# **Google Colab**: the cell below installs missing packages automatically.

import importlib, subprocess, sys

_IMPORT_MAP = {
    "scikit-learn": "sklearn",
}

def _ensure(*pkgs):
    for p in pkgs:
        mod = p.split(">=")[0].split("[")[0]
        mod = _IMPORT_MAP.get(mod, mod).replace("-", "_")
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

_ensure("evidently>=0.5", "scikit-learn>=1.4", "pandas>=2.0", "matplotlib>=3.8", "seaborn>=0.13", "mlflow>=2.10")
print("\u2705 Setup complete")

# ## Imports

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # evidently >= 0.6
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:  # evidently 0.5.x
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

import mlflow
from mlflow.models import infer_signature

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")
print("\u2705 All imports loaded")

# ## Production Concepts Reference
#
# | Concept | What It Means | Why It Matters |
# |---|---|---|
# | **Reproducible Pipeline** | Preprocessing + model bundled as a single artifact | Eliminates training-serving skew |
# | **Training-Serving Skew** | Feature logic differs between training and inference | Silent model degradation |
# | **Model Serialization** | Saving model artifact to disk (joblib, pickle, ONNX) | Enables deployment without retraining |
# | **Data Validation** | Automated checks on schema, types, ranges, nulls | Catches data issues before they reach the model |
# | **Covariate Shift** | Input distribution $P(X)$ changes | Model sees unfamiliar inputs |
# | **Label Shift** | Target distribution $P(Y)$ changes | Class balance no longer matches training |
# | **Concept Drift** | Mapping $P(Y \mid X)$ changes | The relationship the model learned is wrong |
# | **Experiment Tracking** | Record params, metrics, artifacts per run | Reproducibility, comparison, auditing |
# | **Canary Release** | Deploy to a small traffic fraction first | Limits blast radius of failures |
# | **Blue-Green Deployment** | Two parallel environments, instant switch | Fast rollback on failure |
# | **Model Serving** | Expose model as API (REST / gRPC) | Enables real-time inference for applications |
# | **Monitoring** | Continuous tracking of model + data health | Detect degradation before business damage |

# ## Shared Helper Functions
#
# Run the cell below to load all helper functions used throughout this session.

# ── Data Loading ────────────────────────────────────────────────────────

NUMERIC_FEATURES = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
CATEGORICAL_FEATURES = ["workclass", "marital-status", "occupation", "race", "sex"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_adult_data(test_size=0.3, random_state=42):
    """Load Adult Income dataset and return train/test splits."""
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame[ALL_FEATURES + ["class"]].dropna()
    X = df[ALL_FEATURES]
    y = (df["class"].str.contains(">50K")).astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ── Drift Simulation ───────────────────────────────────────────────────

def simulate_covariate_shift(X, column="age", shift=15):
    """Shift a numeric column to simulate covariate drift."""
    X_shifted = X.copy()
    X_shifted[column] = X_shifted[column] + shift
    return X_shifted


def simulate_label_shift(y, flip_rate=0.3, random_state=42):
    """Randomly flip a fraction of labels to simulate label shift."""
    y_shifted = y.copy()
    rng = np.random.RandomState(random_state)
    mask = rng.random(len(y)) < flip_rate
    y_shifted.iloc[mask] = 1 - y_shifted.iloc[mask]
    return y_shifted


def simulate_concept_drift(X, y, feature="hours-per-week", threshold=40):
    """Flip labels for a subgroup to simulate concept drift."""
    y_drifted = y.copy()
    mask = X[feature].values > threshold
    y_drifted.iloc[mask] = 1 - y_drifted.iloc[mask]
    return y_drifted


def simulate_gradual_drift(X_ref, n_windows=10, max_shift=20):
    """Create *n_windows* batches with linearly increasing age shift."""
    windows = []
    for i in range(n_windows):
        shift = max_shift * (i / max(n_windows - 1, 1))
        X_w = X_ref.copy()
        X_w["age"] = X_w["age"] + shift
        noise = np.random.RandomState(i).normal(0, i * 0.5, len(X_w))
        X_w["hours-per-week"] = X_w["hours-per-week"] + noise
        windows.append(X_w)
    return windows


# ── Visualization ──────────────────────────────────────────────────────

def plot_distribution_comparison(ref, prod, columns, title="Reference vs Production"):
    """Overlapping histograms for selected numeric columns."""
    n = len(columns)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(columns):
        axes[i].hist(ref[col], bins=30, alpha=0.5, label="Reference",
                     density=True, color="steelblue")
        axes[i].hist(prod[col], bins=30, alpha=0.5, label="Production",
                     density=True, color="salmon")
        axes[i].set_title(col, fontsize=11)
        axes[i].legend(fontsize=9)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_monitoring_dashboard(metrics):
    """2\u00d72 monitoring dashboard over time windows."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ws = range(len(metrics["accuracy"]))

    # Accuracy
    axes[0, 0].plot(ws, metrics["accuracy"], "b-o", ms=5)
    bl = metrics["baseline_accuracy"]
    axes[0, 0].axhline(bl, color="red", ls="--", label="Baseline")
    axes[0, 0].fill_between(ws, [bl * 0.95] * len(ws), [bl * 1.05] * len(ws),
                            alpha=0.1, color="green", label="\u00b15 % band")
    axes[0, 0].set_title("Model Accuracy")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend(fontsize=8)

    # Drift share
    axes[0, 1].plot(ws, metrics["drift_share"], "g-o", ms=5)
    axes[0, 1].axhline(0.5, color="red", ls="--", label="Alert: \u226550 % features drifted")
    axes[0, 1].set_title("Drift Score (share of drifted features)")
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend(fontsize=8)

    # Mean prediction
    axes[1, 0].plot(ws, metrics["mean_prediction"], "m-o", ms=5)
    axes[1, 0].axhline(metrics["baseline_mean_pred"], color="red", ls="--", label="Baseline")
    axes[1, 0].set_title("Mean Prediction Score")
    axes[1, 0].set_ylabel("P(>50K)")
    axes[1, 0].legend(fontsize=8)

    # Volume
    axes[1, 1].plot(ws, metrics["volume"], "k-o", ms=5)
    axes[1, 1].set_title("Data Volume per Window")
    axes[1, 1].set_ylabel("Samples")

    for ax in axes.flatten():
        ax.set_xlabel("Time Window")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Production Monitoring Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


print("\u2705 Helper functions loaded")

# ## 1. Dataset and Baseline Model
#
# We use the **Adult Income** dataset (predict whether income exceeds \$50K).
#
# This is the same dataset from the lecture example.
# In production, this would be the model your team has already trained and wants to deploy.

X_train, X_test, y_train, y_test = load_adult_data()

print(f"Training samples: {len(X_train):,}")
print(f"Test samples:     {len(X_test):,}")
print(f"Numeric features: {NUMERIC_FEATURES}")
print(f"Categorical features: {CATEGORICAL_FEATURES}")
print(f"\nClass distribution (train):")
print(y_train.value_counts(normalize=True).rename({0: "\u226450K", 1: ">50K"}).to_string())
X_train.head()

# ──── The "notebook approach" that is NOT production-ready ────

# Manual preprocessing — different code paths for train vs inference
X_train_naive = X_train[NUMERIC_FEATURES].copy()
X_test_naive = X_test[NUMERIC_FEATURES].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_naive)
X_test_scaled = scaler.transform(X_test_naive)

naive_model = RandomForestClassifier(n_estimators=100, random_state=42)
naive_model.fit(X_train_scaled, y_train)
naive_acc = naive_model.score(X_test_scaled, y_test)

print(f"Naive model accuracy (numeric features only): {naive_acc:.4f}")
print(f"\n\u26a0\ufe0f Problems with this approach:")
print("  1. Scaler and model are separate objects — easy to mix up")
print("  2. Categorical features are ignored")
print("  3. Need to remember the exact preprocessing steps at inference time")
print("  4. No single artifact to deploy")

# ## 2. Production-Ready Pipeline ✏️ TODO
#
# A production-ready model packages **all preprocessing** and the **estimator** into a single
# `sklearn.Pipeline`. This eliminates training-serving skew.
#
# ### Your task
#
# Build a `ColumnTransformer` + `Pipeline` that:
#
# 1. Applies `SimpleImputer(strategy="median")` + `StandardScaler()` to numeric features
# 2. Applies `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` to categorical features
# 3. Feeds the transformed features into a `RandomForestClassifier(n_estimators=100, random_state=42)`
#
# **Hint**: use `ColumnTransformer` with two named transformers: `"num"` and `"cat"`.

# ✏️ TODO: Build the production pipeline
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", Pipeline([...]), NUMERIC_FEATURES),
#         ("cat", Pipeline([...]), CATEGORICAL_FEATURES),
#     ]
# )
#
# production_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
# ])
#
# production_pipeline.fit(X_train, y_train)
# prod_acc = production_pipeline.score(X_test, y_test)
# print(f"Production pipeline accuracy: {prod_acc:.4f}")
# print(f"Improvement over naive model: {prod_acc - naive_acc:+.4f}")

# ## 3. Model Serialization & Metadata ✏️ TODO
#
# In production you need to know **exactly** what model is running. This means:
#
# - Saving the model artifact (the pipeline)
# - Recording metadata: version, training date, performance metrics, feature list
#
# ### Your task
#
# 1. Save the trained pipeline using `joblib.dump()`
# 2. Create a metadata dictionary with: `model_version`, `trained_at`, `test_accuracy`, `features`, `n_training_samples`
# 3. Save the metadata as JSON alongside the model

# ✏️ TODO: Serialize the model and save metadata
#
# MODEL_DIR = Path("model_artifacts")
# MODEL_DIR.mkdir(exist_ok=True)
#
# model_path = MODEL_DIR / "adult_income_pipeline_v1.pkl"
# joblib.dump(production_pipeline, model_path)
#
# metadata = {
#     "model_version": ...,
#     "trained_at": ...,
#     "test_accuracy": ...,
#     "features": ...,
#     "n_training_samples": ...,
# }
#
# metadata_path = MODEL_DIR / "model_metadata_v1.json"
# with open(metadata_path, "w") as f:
#     json.dump(metadata, f, indent=2)
#
# print(json.dumps(metadata, indent=2))

# Verify: load the artifact in a "clean" context and check it works
loaded_pipeline = joblib.load(model_path)
loaded_acc = loaded_pipeline.score(X_test, y_test)

with open(metadata_path) as f:
    loaded_meta = json.load(f)

print(f"Loaded model version: {loaded_meta['model_version']}")
print(f"Loaded model accuracy: {loaded_acc:.4f}")
assert abs(loaded_acc - prod_acc) < 1e-6, "Accuracy mismatch after reload!"
print("\u2705 Model artifact verified \u2014 same accuracy after serialization")

# ### Why Metadata Matters
#
# Production debugging checklist when something goes wrong:
#
# | Question | Where to find the answer |
# |---|---|
# | Which model version is running? | `model_version` in metadata |
# | When was it trained? | `trained_at` |
# | What accuracy did it have at training time? | `test_accuracy` |
# | What features does it expect? | `features` |
# | How much data was it trained on? | `n_training_samples` |
#
# Without this metadata, diagnosing production issues becomes guesswork.

# ## 4. Experiment Tracking with MLflow
#
# In a notebook, model results live in cell outputs and get lost when the kernel restarts.
# In production, every training run must be **logged and comparable**.
#
# [MLflow](https://mlflow.org/) is the most widely-used open-source experiment tracking platform.
# It records:
#
# - **Parameters** — hyperparameters, feature lists, data versions
# - **Metrics** — accuracy, F1, latency, custom KPIs
# - **Artifacts** — model files, plots, metadata
# - **Tags** — free-form annotations (author, ticket, dataset version)
#
# MLflow runs locally out of the box — no server needed for this demo.

# Set up MLflow to log locally (no server needed — works in Colab)
mlflow.set_tracking_uri("mlite.db")  # local SQLite backend
mlflow.set_experiment("adult_income_production")

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: adult_income_production")
print("\u2705 MLflow configured")

# ✏️ TODO: Log a training run to MLflow
#
# with mlflow.start_run(run_name="rf_baseline_v1") as run:
#     # Log parameters — record what you trained
#     mlflow.log_param("model_type", "RandomForestClassifier")
#     mlflow.log_param("n_estimators", ...)
#     mlflow.log_param("n_training_samples", ...)
#
#     # Log metrics — record how it performed
#     mlflow.log_metric("test_accuracy", prod_acc)
#     mlflow.log_metric("naive_accuracy", naive_acc)
#
#     # Log the model artifact itself
#     signature = infer_signature(X_test, production_pipeline.predict(X_test))
#     mlflow.sklearn.log_model(production_pipeline, "model", signature=signature)
#
#     # Log extra files
#     mlflow.log_artifact(str(metadata_path))
#
#     print(f"Run logged: {run.info.run_id}")

# Compare runs — simulate a second experiment with different hyperparameters
with mlflow.start_run(run_name="rf_more_trees_v2") as run2:
    pipeline_v2 = Pipeline([
        ("preprocessor", production_pipeline.named_steps["preprocessor"]),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
    ])
    pipeline_v2.fit(X_train, y_train)
    acc_v2 = pipeline_v2.score(X_test, y_test)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 15)
    mlflow.log_metric("test_accuracy", acc_v2)

    signature = infer_signature(X_test, pipeline_v2.predict(X_test))
    mlflow.sklearn.log_model(pipeline_v2, "model", signature=signature)

# Retrieve and compare all runs
runs_df = mlflow.search_runs()
comparison = runs_df[["run_id", "tags.mlflow.runName",
                      "params.n_estimators", "metrics.test_accuracy"]].head()
print("Experiment comparison:")
print(comparison.to_string(index=False))
print(f"\n\u2705 Both runs tracked — you can compare, rollback, or promote any version")

# ### Why Experiment Tracking Matters
#
# | Without tracking | With tracking |
# |---|---|
# | "Which model was better?" — scroll through old cells | Query MLflow: sort runs by metric |
# | "What hyperparameters did I use?" — lost | Every parameter logged and searchable |
# | "Can we reproduce last week's result?" — maybe | Artifact + params + data hash = reproducible |
# | "Who trained this and when?" — unclear | Run metadata with timestamps and tags |
#
# > In a team setting, MLflow can run as a **shared server** so all team members see each other's experiments.
# > For this demo we use a local backend, but the API is identical.

# ## 5. MLOps Tool Landscape
#
# MLflow is one of many tools in the MLOps ecosystem. This section gives you a map
# of the most important categories and tools so you know **what exists and when to use it**.
#
# ### Experiment Tracking & Model Registry
#
# | Tool | Key Strengths | Setup | Best For |
# |---|---|---|---|
# | [**MLflow**](https://mlflow.org/) | Open-source, self-hosted, broad integrations | `pip install mlflow` — no signup | Teams wanting full control, on-prem |
# | [**Weights & Biases (W&B)**](https://wandb.ai/) | Beautiful dashboards, collaboration, sweeps | `pip install wandb` — free tier with signup | Research teams, hyperparameter search |
# | [**ClearML**](https://clear.ml/) | Auto-logging, orchestration, data management | `pip install clearml` — free tier | End-to-end MLOps with minimal code changes |
# | [**Neptune.ai**](https://neptune.ai/) | Flexible metadata store, custom dashboards | `pip install neptune` — free tier | Large-scale experiment comparison |
#
# ### Data & Model Monitoring
#
# | Tool | Key Strengths | Best For |
# |---|---|---|
# | [**Evidently**](https://github.com/evidentlyai/evidently) | Drift reports, test suites, open-source | Data drift detection, model quality monitoring |
# | [**WhyLabs / whylogs**](https://github.com/whylabs/whylogs) | Lightweight profiling, streaming support | Real-time data quality monitoring |
# | [**NannyML**](https://github.com/NannyML/nannyml) | Performance estimation without labels | Monitoring when ground truth is delayed |
#
# ### Model Serving & Deployment
#
# | Tool | Key Strengths | Best For |
# |---|---|---|
# | [**BentoML**](https://github.com/bentoml/BentoML) | Python-native, Dockerized serving, batching | Packaging models as production APIs |
# | [**MLflow Models**](https://mlflow.org/docs/latest/ml/deployment/) | Built into MLflow, multiple flavors | Simple deployment of tracked models |
# | [**Seldon Core**](https://github.com/SeldonIO/seldon-core) | Kubernetes-native, A/B testing, canary | Enterprise Kubernetes deployments |
# | [**TorchServe**](https://github.com/pytorch/serve) / [**TF Serving**](https://github.com/tensorflow/serving) | Framework-optimized inference | High-throughput DL model serving |
#
# ### Pipeline Orchestration
#
# | Tool | Key Strengths | Best For |
# |---|---|---|
# | [**Apache Airflow**](https://airflow.apache.org/) | Industry standard, DAG-based workflows | Scheduled batch pipelines |
# | [**Prefect**](https://www.prefect.io/) | Modern Python-native, dynamic flows | ML pipelines with complex dependencies |
# | [**Kubeflow**](https://www.kubeflow.org/) | Kubernetes-native, end-to-end ML | Full ML platform on Kubernetes |
# | [**ZenML**](https://zenml.io/) | MLOps framework, tool-agnostic | Connecting existing tools into pipelines |
#
# ### Data Versioning
#
# | Tool | Key Strengths | Best For |
# |---|---|---|
# | [**DVC**](https://dvc.org/) | Git-like interface for data, pipeline tracking | Versioning datasets alongside code |
# | [**LakeFS**](https://lakefs.io/) | Git-like operations on data lakes | Large-scale data versioning |

# ### BentoML: From Model to API
#
# [BentoML](https://github.com/bentoml/BentoML) solves the "last mile" problem:
# how to turn a trained model into a **production API** that other services can call.
#
# The typical BentoML workflow:
#
# ```
# 1. Save model → bentoml.sklearn.save_model("income_model", pipeline)
# 2. Define service → service.py with @bentoml.api decorator
# 3. Build → bentoml build (creates a Docker-ready artifact)
# 4. Serve → bentoml serve service:IncomePredictionService
# ```
#
# Below is what a BentoML service definition looks like.
# We write it to a file — you can examine it, but we do not actually start the server in this notebook.

# Write a BentoML service definition (conceptual — read and understand)

service_code = '''
import bentoml
import numpy as np
import pandas as pd


@bentoml.service(name="income_prediction_service")
class IncomePredictionService:
    \"\"\"Production API for the Adult Income model.\"\"\"

    def __init__(self):
        # Load the model saved with bentoml.sklearn.save_model()
        self.model = bentoml.sklearn.load_model("income_classifier:latest")

    @bentoml.api()
    def predict(self, input_data: dict) -> dict:
        \"\"\"Predict income class from raw features.\"\"\"
        df = pd.DataFrame([input_data])
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0].tolist()
        return {
            "prediction": int(prediction),
            "label": ">50K" if prediction == 1 else "<=50K",
            "probability": probability,
        }

    @bentoml.api()
    def healthcheck(self) -> dict:
        \"\"\"Health check endpoint for monitoring.\"\"\"
        return {"status": "healthy", "model": "income_classifier"}
'''

print("\U0001f4c4 BentoML service definition:")
print(service_code)
print("\u2192 This file would be served with: bentoml serve service:IncomePredictionService")
print("\u2192 BentoML automatically generates a REST API + Swagger docs")
print("\u2192 bentoml build creates a Docker image ready for deployment")

# ### Weights & Biases (W&B) vs ClearML — Quick Demo Comparison
#
# Both W&B and ClearML provide experiment tracking similar to MLflow but with different strengths.
# Below is pseudocode showing how logging looks in each — the concepts are the same.
#
# **Weights & Biases:**
# ```python
# import wandb
#
# wandb.init(project="adult-income", name="rf-baseline")
# wandb.config.update({"n_estimators": 100, "model": "RandomForest"})
# wandb.log({"accuracy": 0.856, "f1": 0.72})
# wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(y_true, y_pred)})
# wandb.finish()
# ```
#
# **ClearML:**
# ```python
# from clearml import Task
#
# task = Task.init(project_name="adult-income", task_name="rf-baseline")
# task.connect({"n_estimators": 100, "model": "RandomForest"})
# # ClearML auto-captures matplotlib plots, prints, and git info!
# import matplotlib.pyplot as plt
# plt.plot(...)  # automatically logged
# ```
#
# > **Key difference**: W&B excels at visualizations and team collaboration.
# > ClearML excels at auto-capture (less code needed) and built-in orchestration.
# > MLflow excels at self-hosting and model registry.
#
# ### How to Choose?
#
# | Factor | MLflow | W&B | ClearML |
# |---|---|---|---|
# | **Hosting** | Self-hosted (full control) | Cloud (managed) | Both (cloud + self-hosted) |
# | **Setup effort** | Minimal (local file) | Signup + API key | Signup + API key |
# | **Auto-logging** | Framework-specific | Broad | Very broad (matplotlib, stdout) |
# | **Collaboration** | Manual sharing | Built-in dashboards | Built-in dashboards |
# | **Pricing** | Free (open-source) | Free tier, paid for teams | Free tier, paid for teams |
# | **Orchestration** | Separate (Airflow, etc.) | Sweeps (HPO) | Built-in pipelines + HPO |
# | **Model serving** | MLflow Models | W&B Launch | ClearML Serving |

# ## 6. Data Validation Gates ✏️ TODO
#
# Before feeding data to a model, production systems validate it. This catches issues like:
#
# - Missing columns (schema change upstream)
# - Unexpected null values
# - Values outside expected ranges
# - New categorical values never seen in training
#
# ### Your task
#
# Implement a `validate_data(df, schema)` function that checks:
#
# 1. All expected columns are present
# 2. Numeric columns have no more than 10 % null values
# 3. Numeric columns are within expected ranges
# 4. Return a list of validation errors (empty list = all checks passed)

# ✏️ TODO: Implement data validation

SCHEMA = {
    "expected_columns": ALL_FEATURES,
    "numeric_ranges": {
        "age": (17, 90),
        "education-num": (1, 16),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    },
    "max_null_rate": 0.10,
}


def validate_data(df, schema=SCHEMA):
    """Validate incoming data against expected schema. Return list of errors."""
    errors = []

    # TODO: Check that all expected columns are present
    # missing = set(schema["expected_columns"]) - set(df.columns)
    # if missing: errors.append(...)

    # TODO: Check null rates for each column
    # for col in schema["expected_columns"]:
    #     ...

    # TODO: Check numeric ranges
    # for col, (lo, hi) in schema["numeric_ranges"].items():
    #     ...

    return errors

# Test on clean data
errors_clean = validate_data(X_test)
print("Clean data validation:")
if not errors_clean:
    print("  \u2705 All checks passed")
else:
    for e in errors_clean:
        print(f"  \u274c {e}")

# Test on corrupted data
X_corrupted = X_test.copy()
X_corrupted["age"] = X_corrupted["age"] + 50            # push ages out of range
X_corrupted.loc[X_corrupted.index[:1000], "workclass"] = np.nan  # inject nulls
X_corrupted = X_corrupted.drop(columns=["sex"])          # remove a column

print("\nCorrupted data validation:")
errors_bad = validate_data(X_corrupted)
for e in errors_bad:
    print(f"  \u274c {e}")
print(f"\n\u2192 {len(errors_bad)} validation errors caught before the model saw any data")

# ### Interpretation Guide
#
# 1. What would happen if corrupted data reached the model without validation?
# 2. Which of these errors would cause a *silent* failure (wrong predictions) vs an *explicit* crash?
# 3. In a real production system, who should be notified when validation fails?
# 4. How often should the schema be updated?

# ## 7. Simulating Production Data & Drift Types
#
# In production, data changes over time. The lecture identifies three types of drift:
#
# - **Covariate shift**: $P(X)$ changes — the input distribution shifts
# - **Label shift**: $P(Y)$ changes — the class balance shifts
# - **Concept drift**: $P(Y \mid X)$ changes — the relationship between features and target changes
#
# We simulate all three and measure how they affect the model.

# Simulate three types of drift
X_covariate = simulate_covariate_shift(X_test, column="age", shift=15)
y_label = simulate_label_shift(y_test, flip_rate=0.3)
y_concept = simulate_concept_drift(X_test, y_test, feature="hours-per-week", threshold=40)

# Measure impact on model performance
acc_original = accuracy_score(y_test, production_pipeline.predict(X_test))
acc_covariate = accuracy_score(y_test, production_pipeline.predict(X_covariate))
acc_label = accuracy_score(y_label, production_pipeline.predict(X_test))
acc_concept = accuracy_score(y_concept, production_pipeline.predict(X_test))

results = pd.DataFrame({
    "Scenario": ["No drift", "Covariate shift (+15 age)",
                  "Label shift (30% flipped)", "Concept drift (hours>40)"],
    "Accuracy": [acc_original, acc_covariate, acc_label, acc_concept],
    "\u0394 Accuracy": [0, acc_covariate - acc_original,
                        acc_label - acc_original, acc_concept - acc_original],
})
results.style.format({"Accuracy": "{:.4f}", "\u0394 Accuracy": "{:+.4f}"}).background_gradient(
    subset=["\u0394 Accuracy"], cmap="RdYlGn", vmin=-0.15, vmax=0.05,
)

plot_distribution_comparison(
    X_test, X_covariate,
    columns=NUMERIC_FEATURES,
    title="Covariate Shift: Reference vs Production (age +15)",
)

# ### Drift Type Recognition Guide
#
# | Drift Type | What Changed | How to Detect | Impact |
# |---|---|---|---|
# | **Covariate** | Input features $P(X)$ | Distribution tests (KS, PSI) on features | Model sees unfamiliar inputs |
# | **Label** | Target $P(Y)$ | Class balance monitoring | Predictions may become miscalibrated |
# | **Concept** | Mapping $P(Y \mid X)$ | Performance monitoring (accuracy, F1) | Model's learned rules are wrong |
#
# > **Key insight**: Covariate shift is detectable WITHOUT labels. Concept drift requires labels to confirm.

# ## 8. Drift Detection with Evidently ✏️ TODO
#
# [Evidently](https://github.com/evidentlyai/evidently) is an open-source library for data and ML monitoring.
# It provides rich visual reports and statistical tests for drift detection.

# Run Evidently data drift report: reference (train) vs production (covariate shift)
ref_df = X_train[NUMERIC_FEATURES].reset_index(drop=True)
prod_df = X_covariate[NUMERIC_FEATURES].reset_index(drop=True)

drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=ref_df, current_data=prod_df)
drift_report

# Under the hood: Kolmogorov-Smirnov test for each feature
print("Manual drift detection (KS test per feature):")
print(f"{'Feature':<20} {'KS Statistic':>12} {'p-value':>12} {'Drifted?':>10}")
print("-" * 56)

n_drifted = 0
for col in NUMERIC_FEATURES:
    stat, pval = ks_2samp(X_train[col], X_covariate[col])
    drifted = pval < 0.05
    n_drifted += drifted
    marker = "\u26a0\ufe0f YES" if drifted else "  no"
    print(f"{col:<20} {stat:>12.4f} {pval:>12.6f} {marker:>10}")

print(f"\n\u2192 {n_drifted}/{len(NUMERIC_FEATURES)} features show statistically significant drift")

# ### Your Turn: Detect Concept Drift with Evidently ✏️ TODO
#
# Concept drift is harder to detect from features alone because $P(X)$ may not change.
#
# **Tasks**:
# 1. Run an Evidently drift report comparing test data (no covariate shift) against training data
# 2. Run KS tests on features
# 3. Compare model accuracy on concept-drifted labels vs original labels
# 4. Answer: **Can you detect concept drift from input features alone?**

# ✏️ TODO: Run drift detection on concept drift scenario
#
# 1. Run Evidently DataDriftPreset on X_test (numeric only) vs X_train (numeric only)
#    Remember: concept drift changed Y but NOT X
#
# concept_drift_report = Report(metrics=[DataDriftPreset()])
# concept_drift_report.run(
#     reference_data=...,  # training data
#     current_data=...,    # production data
# )
# concept_drift_report

# ✏️ TODO: Analyze why concept drift is invisible in features
#
# 1. Run KS tests on NUMERIC_FEATURES: compare X_train vs X_test
# 2. Print accuracy on original vs concept-drifted labels
# 3. Explain: why do features look clean while accuracy dropped?
#
# for col in NUMERIC_FEATURES:
#     _, pval = ks_2samp(X_train[col], X_test[col])
#     print(f"{col}: p-value={pval:.6f}")
#
# print(f"Accuracy on original labels:       {acc_original:.4f}")
# print(f"Accuracy on concept-drifted labels: {acc_concept:.4f}")

# ### Interpretation Guide
#
# 1. Why did the Evidently report show no drift for concept drift, even though accuracy dropped?
# 2. In production, how would you detect concept drift if labels arrive with a delay?
# 3. What proxy signals could indicate concept drift even before labels are available?
# 4. Why is this the hardest type of drift to handle in production?

# ## 9. Monitoring Over Time ✏️ TODO
#
# In production, drift does not happen all at once. It builds up gradually.
#
# A monitoring system tracks metrics over **time windows** and alerts when thresholds are breached.
#
# We simulate 10 production time windows with gradually increasing drift and build a monitoring dashboard.

# Simulate 10 production windows with gradually increasing drift
windows = simulate_gradual_drift(X_test, n_windows=10, max_shift=20)

metrics = {
    "accuracy": [],
    "drift_share": [],
    "mean_prediction": [],
    "volume": [],
    "baseline_accuracy": acc_original,
    "baseline_mean_pred": production_pipeline.predict_proba(X_test)[:, 1].mean(),
}

for i, X_w in enumerate(windows):
    y_pred = production_pipeline.predict(X_w)
    metrics["accuracy"].append(accuracy_score(y_test, y_pred))

    n_d = sum(
        1 for col in NUMERIC_FEATURES
        if ks_2samp(X_train[col], X_w[col]).pvalue < 0.05
    )
    metrics["drift_share"].append(n_d / len(NUMERIC_FEATURES))

    proba = production_pipeline.predict_proba(X_w)[:, 1]
    metrics["mean_prediction"].append(proba.mean())
    metrics["volume"].append(len(X_w))

plot_monitoring_dashboard(metrics)

# ✏️ TODO: Identify the retraining trigger point
#
# Look at the dashboard above and determine:
# 1. At which time window does accuracy drop below baseline by more than 2%?
# 2. At which window does drift share exceed 50%?
# 3. Which signal fires first — accuracy drop or drift detection?
#
# accuracy_threshold = metrics["baseline_accuracy"] * 0.98
# drift_threshold = 0.5
#
# acc_trigger = None
# drift_trigger = None
# for i in range(len(windows)):
#     if acc_trigger is None and metrics["accuracy"][i] < accuracy_threshold:
#         acc_trigger = i
#     if drift_trigger is None and metrics["drift_share"][i] >= drift_threshold:
#         drift_trigger = i
#
# print(f"Accuracy trigger at window: {acc_trigger}")
# print(f"Drift trigger at window:    {drift_trigger}")

# ### Interpretation Guide
#
# 1. Is drift detection a leading or lagging indicator compared to accuracy monitoring?
# 2. What are the trade-offs of setting a lower drift threshold (more sensitive)?
# 3. In practice, if labels are delayed by days or weeks, which monitoring signals can you still use?
# 4. When should you retrain vs investigate the root cause first?

# ## 10. Deployment Strategy Simulation ✏️ TODO
#
# The lecture describes several deployment patterns. We simulate the most common one:
# **canary release**.
#
# ### How canary release works
#
# 1. Deploy the new model to a **small fraction** of traffic (e.g. 5 %)
# 2. Monitor key metrics on the canary traffic
# 3. If metrics are healthy → gradually increase traffic
# 4. If metrics degrade → **rollback** to the old model
#
# This limits the "blast radius" of a bad deployment.

# ✏️ TODO: Implement canary release decision logic
#
# def canary_release(old_model, new_model, X, y,
#                    canary_fraction=0.05, accuracy_drop_limit=0.01):
#     """
#     Simulate a canary release.
#     Route canary_fraction of traffic to new_model, compare with old_model.
#     Return decision dict with 'decision' ('PROMOTE' or 'ROLLBACK') and 'reason'.
#     """
#     n_canary = max(int(len(X) * canary_fraction), 50)
#     rng = np.random.RandomState(42)
#     idx = rng.choice(len(X), size=n_canary, replace=False)
#     X_canary = X.iloc[idx]
#     y_canary = y.iloc[idx]
#
#     acc_old = accuracy_score(y_canary, old_model.predict(X_canary))
#     acc_new = accuracy_score(y_canary, new_model.predict(X_canary))
#     delta = acc_new - acc_old
#
#     # TODO: Decide — PROMOTE or ROLLBACK?
#     # If delta >= -accuracy_drop_limit → PROMOTE, else → ROLLBACK
#
#     return {"decision": ..., "reason": ..., "canary_size": n_canary,
#             "accuracy_old": acc_old, "accuracy_new": acc_new, "delta": delta}

# Scenario 1: Retrained model (more trees, different seed) — should pass
retrained = Pipeline([
    ("preprocessor", production_pipeline.named_steps["preprocessor"]),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=99)),
])
retrained.fit(X_train, y_train)

result_good = canary_release(production_pipeline, retrained, X_test, y_test)
print("Scenario 1: Retrained model (more trees, different seed)")
print(f"  Decision: {result_good['decision']}")
print(f"  {result_good['reason']}")
print(f"  Canary size: {result_good['canary_size']}\n")

# Scenario 2: Model trained on corrupted data — should rollback
X_bad = X_train.copy()
X_bad["age"] = np.random.RandomState(0).randint(0, 100, len(X_bad))
X_bad["hours-per-week"] = np.random.RandomState(1).randint(0, 100, len(X_bad))
broken_pipeline = Pipeline([
    ("preprocessor", production_pipeline.named_steps["preprocessor"]),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])
broken_pipeline.fit(X_bad, y_train)

result_bad = canary_release(production_pipeline, broken_pipeline, X_test, y_test)
print("Scenario 2: Model trained on corrupted data")
print(f"  Decision: {result_bad['decision']}")
print(f"  {result_bad['reason']}")
print(f"  Canary size: {result_bad['canary_size']}")
print("\n\U0001f4a1 The canary caught the bad model before it reached all users!")

# ### Interpretation Guide
#
# 1. Why is the canary fraction typically small (1–5 %)?
# 2. What metrics beyond accuracy should a canary check monitor?
# 3. How would you decide the `accuracy_drop_limit` in a real project?
# 4. What is the trade-off between canary sample size and detection speed?
# 5. How does canary release compare to A/B testing?

# ## 11. Production Readiness Checklist ✏️ TODO
#
# Before deploying any model, teams should complete a structured readiness review.
#
# Fill in the checklist below based on what we built in this session.
# Mark each item as ✅ (done), ⚠️ (partial), or ❌ (not done).

# ✏️ TODO: Fill in the checklist based on our session work
# Change the status and evidence for each item.

checklist = {
    "Reproducible Pipeline": {
        "status": "?",
        "evidence": "TODO",
    },
    "Model Serialization": {
        "status": "?",
        "evidence": "TODO",
    },
    "Model Metadata & Versioning": {
        "status": "?",
        "evidence": "TODO",
    },
    "Data Validation": {
        "status": "?",
        "evidence": "TODO",
    },
    "Experiment Tracking": {
        "status": "?",
        "evidence": "TODO",
    },
    "Drift Detection": {
        "status": "?",
        "evidence": "TODO",
    },
    "Performance Monitoring": {
        "status": "?",
        "evidence": "TODO",
    },
    "Model Serving": {
        "status": "?",
        "evidence": "TODO",
    },
    "Deployment Strategy": {
        "status": "?",
        "evidence": "TODO",
    },
    "Retraining Policy": {
        "status": "?",
        "evidence": "TODO",
    },
    "CI/CD Pipeline": {
        "status": "?",
        "evidence": "TODO",
    },
    "Fairness & Responsible AI": {
        "status": "?",
        "evidence": "TODO",
    },
}

print("Production Readiness Checklist")
print("=" * 60)
for item, detail in checklist.items():
    print(f"\n{detail['status']} {item}")
    print(f"   \u2192 {detail['evidence']}")

done = sum(1 for d in checklist.values() if d["status"] == "\u2705")
partial = sum(1 for d in checklist.values() if d["status"] == "\u26a0\ufe0f")
missing = sum(1 for d in checklist.values() if d["status"] == "\u274c")
print(f"\n{'=' * 60}")
print(f"Score: {done} done, {partial} partial, {missing} not covered")

# ## 12. Debrief
#
# ### Key Takeaways
#
# 1. **A trained model \u2260 a production model.** Production requires pipelines, serialization,
#    validation, monitoring, and deployment strategy.
#
# 2. **Training-serving skew** is eliminated by packaging preprocessing into the pipeline.
#    A `ColumnTransformer` + `Pipeline` is the minimum viable production artifact.
#
# 3. **Data validation** is a cheap safety net.
#    Catching schema and quality issues before inference prevents silent failures.
#
# 4. **Not all drift is the same.** Covariate shift is detectable from features alone.
#    Concept drift requires labels — and labels are often delayed.
#
# 5. **Monitoring is not optional.** Without it, models degrade silently.
#    Drift detection is a *leading* indicator that can fire before accuracy drops.
#
# 6. **Canary releases limit blast radius.**
#    Never deploy a new model to 100 % of traffic at once in a system that matters.
#
# 7. **The MLOps ecosystem is large but navigable.**
#    MLflow, W&B, ClearML for tracking; Evidently for monitoring; BentoML for serving.
#    You don't need all of them — pick tools that match your team's needs.
#
# ### Scope Note
#
# This 90-minute session covered hands-on production building blocks:
# pipelines, serialization, experiment tracking, validation, drift detection,
# monitoring, deployment logic, and a tour of the MLOps tool landscape.
#
# Topics from the lecture that are important but require infrastructure beyond a notebook:
#
# - Full CI/CD for ML (GitHub Actions, Jenkins)
# - Container-based deployment (Docker, Kubernetes)
# - Feature stores and online/offline feature consistency
# - A/B testing with real traffic routing
# - Infrastructure autoscaling and cost optimization
# - Stakeholder communication and team dynamics
# - W&B Sweeps / ClearML HPO (require account setup)
#
# ### Recommended Further Reading
#
# **Experiment Tracking & Model Registry:**
# - [MLflow — Open-Source ML Platform](https://mlflow.org/)
# - [Weights & Biases (W&B) — Experiment Tracking](https://wandb.ai/)
# - [ClearML — End-to-End MLOps](https://clear.ml/)
# - [Neptune.ai — Metadata Store for ML](https://neptune.ai/)
#
# **Model Serving & Deployment:**
# - [BentoML — Build Production-Ready AI APIs](https://github.com/bentoml/BentoML)
# - [Seldon Core — Kubernetes Model Serving](https://github.com/SeldonIO/seldon-core)
#
# **Monitoring & Data Quality:**
# - [Evidently AI — Open-Source ML Monitoring](https://github.com/evidentlyai/evidently)
# - [whylogs — Data Logging & Profiling](https://github.com/whylabs/whylogs)
# - [NannyML — Performance Estimation Without Labels](https://github.com/NannyML/nannyml)
#
# **Pipeline Orchestration & Data Versioning:**
# - [DVC — Data Version Control](https://dvc.org/)
# - [Apache Airflow — Workflow Orchestration](https://airflow.apache.org/)
# - [ZenML — MLOps Framework](https://zenml.io/)
#
# **Foundational Papers & Guides:**
# - [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
# - [Sculley et al. — Hidden Technical Debt in ML Systems (2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
# - [Made With ML — MLOps Course](https://madewithml.com/)

