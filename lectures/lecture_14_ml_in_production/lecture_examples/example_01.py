# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # ML in Production - Guided Practice
# 
# Welcome to the guided practice session on **Machine Learning in Production**.
# 
# In this notebook, we focus on three practical building blocks:
# 1. **Reproducible preprocessing** with a `Pipeline`
# 2. **Model serialization** for handoff and deployment
# 3. **Simple drift detection** to monitor production data
# 
# ---

# %% [markdown]
# ## 1. Load and Prepare a Simple Production Dataset
# 
# We reuse the Adult Income dataset to simulate a small production workflow.

# %%
import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("Loading Adult Income dataset...")
data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
df = data.frame

features = ["age", "education-num", "hours-per-week", "sex", "class"]
df = df[features].dropna()
df["sex_encoded"] = df["sex"].map({"Male": 1, "Female": 0})
df["target"] = df["class"].apply(lambda value: 1 if ">50K" in str(value) else 0)

X = df[["age", "education-num", "hours-per-week", "sex_encoded"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# %% [markdown]
# ## 2. Build a Reproducible Inference Pipeline
# 
# Production systems should package preprocessing and the estimator together.

# %%
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print(f"Pipeline test accuracy: {test_accuracy:.4f}")

# %% [markdown]
# ## 3. Serialize the Model Artifact
# 
# Export the trained pipeline so that the same artifact can be reused later.

# %%
model_path = "adult_income_pipeline_v1.pkl"
joblib.dump(pipeline, model_path)
print(f"Saved model artifact to {model_path}")

# %% [markdown]
# ## 4. Simulate and Detect Data Drift
# 
# We create a synthetic production slice with an older population profile and compare it with training data.

# %%
production_slice = X_test.copy()
production_slice["age"] = production_slice["age"] + 15

statistic, p_value = ks_2samp(X_train["age"], production_slice["age"])

print(f"KS statistic: {statistic:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("Drift detected in the age feature.")
else:
    print("No statistically significant drift detected.")

# %% [markdown]
# ## 5. Deployment Checklist
# 
# Before moving from notebook to service, confirm:
# 
# - preprocessing is bundled into the exported pipeline
# - the saved artifact can be loaded in a clean environment
# - input schema is fixed and validated
# - drift checks run on a schedule
# - metrics and alerts are wired before rollout
