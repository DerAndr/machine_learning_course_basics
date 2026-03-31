# /// script
# source-notebook = "example_02.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]

# %%
# NOTE: notebook magic commented for local script use: ! pip install shap

# %%
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt

# Step 1: Load Diabetes Dataset
diabetes_data = load_diabetes()
X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = diabetes_data.target

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 4: Explainability with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot for Diabetes Data")
plt.show()

# SHAP Dependence Plot (example: 'bmi' - Body Mass Index)
shap.dependence_plot('bmi', shap_values, X_test)

# Select a specific test instance
i = 0  # Index of the instance to explain
shap.force_plot(
    explainer.expected_value[0],
    shap_values[i],
    X_test.iloc[i],
    matplotlib=True,
    text_rotation=45
)

# %% [markdown]
# SHAP summary plot provides a global view of feature importance and how each feature influences predictions:
# 
# * Y-axis: Features sorted by importance (highest at the top).
# * X-axis: SHAP values (impact on model output).
# * Color: Represents the feature value (red for high values, blue for low values).
# 
# Interpretation:
# 
# Feature importance: bmi has the highest overall importance in predicting diabetes progression, followed by s5 and bp.
# Direction of impact:
# High values of bmi (red) have a positive impact on predictions (increase the target value).
# Low values of bmi (blue) have a negative impact.
# Feature interactions: The spread in SHAP values for each feature indicates how its impact varies across different data points.
# 
# 
# ---
# 

# %% [markdown]
# SHAP dependence plot for the feature bmi (Body Mass Index) provides the following insights:
# 
# * X-axis (bmi): The actual values of the bmi feature.
# * Y-axis (SHAP value for bmi): The contribution of bmi to the model's prediction.
# * Color (s2): The color represents the values of another feature (s2), showing interactions between bmi and s2.
# 
# Interpretation:
# 
# Higher bmi values generally lead to higher SHAP values, indicating a positive relationship between bmi and the target (diabetes progression).
# The interaction with s2 shows that its value further modulates the contribution of bmi (e.g., higher s2 values tend to amplify the SHAP values for high bmi).

# %%
# NOTE: notebook magic commented for local script use: ! pip install lime

# %%
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lime.lime_tabular
import matplotlib.pyplot as plt

# Step 1: Load Diabetes Dataset
diabetes_data = load_diabetes()
X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = diabetes_data.target
feature_names = X.columns

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 4: Explainability with LIME
# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    mode="regression",
    verbose=True
)

# Choose a specific instance from the test set to explain
i = 10  # Index of the instance to explain
instance = X_test.iloc[i].values.reshape(1, -1)

# Generate explanation
exp = explainer.explain_instance(
    data_row=instance.flatten(),
    predict_fn=model.predict
)

# Display explanation in text form
print("\nLIME Explanation for Instance:")
for feature, value in exp.as_list():
    print(f"{feature}: {value:.3f}")

# Generate and display the force plot-like visualization
exp.show_in_notebook(show_table=True)

# %%
# NOTE: notebook magic commented for local script use: !pip install alibi

# %%
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import matplotlib.pyplot as plt
from alibi.explainers import ALE, plot_ale

# Step 1: Load Diabetes Dataset
diabetes_data = load_diabetes()
X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = diabetes_data.target
feature_names = X.columns

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Feature Importance from Random Forest
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print("Feature Importance:")
print(importance_df)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], edgecolor='k')
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# Step 5: Partial Dependence Plots (PDP)
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 5], feature_names=feature_names, ax=ax
)
plt.suptitle("Partial Dependence Plots")
plt.tight_layout()
plt.show()

# Step 6: Accumulated Local Effects (ALE)
ale_explainer = ALE(model.predict, feature_names=feature_names)
ale_exp = ale_explainer.explain(X_train.values)

# ALE Plot for 'bmi'
fig, ax = plt.subplots(figsize=(8, 6))
plot_ale(ale_exp, features=["bmi"], ax=ax)
plt.title("ALE Plot for 'bmi'")
plt.tight_layout()
plt.show()

# Step 7: Permutation Importance
perm_importance = permutation_importance(
    model, X_test, y_test, n_repeats=30, random_state=42, scoring="neg_mean_squared_error"
)

# Convert permutation importance to DataFrame
perm_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm_importance.importances_mean,
    "StdDev": perm_importance.importances_std
}).sort_values(by="Importance", ascending=False)

print("Permutation Importance:")
print(perm_df)

# Plot Permutation Importance
plt.figure(figsize=(10, 6))
plt.barh(perm_df["Feature"], perm_df["Importance"], xerr=perm_df["StdDev"], edgecolor='k')
plt.xlabel("Permutation Importance")
plt.title("Permutation Importance with Error Bars")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# # Key Components of the ALE Plot
# 
# ## X-axis (`bmi`):
# - Represents the values of the `bmi` feature (Body Mass Index) in the dataset.
# 
# ## Y-axis (`ALE`):
# - Shows the accumulated local effect of the `bmi` feature on the model's prediction. This indicates how the model's predictions change as the value of `bmi` changes.
# 
# ## Zero Line:
# - The horizontal line at `ALE = 0` represents the baseline prediction of the model (no influence from `bmi`).
#   - Positive ALE values mean the feature increases the model's prediction.
#   - Negative ALE values mean the feature decreases the model's prediction.
# 
# ## Vertical Ticks:
# - Represent the distribution of `bmi` values in the training data.
#   - Sparse ticks indicate less common `bmi` values, while dense regions indicate frequently observed values.
# 
# ---
# 
# # Interpreting the Plot
# 
# ## Impact of `bmi` on Predictions:
# - For `bmi` values below approximately `-0.05`, the ALE is negative, meaning these values decrease the model's prediction compared to the baseline.
# - Between `-0.05` and `0.05`, the ALE fluctuates around 0, showing a mix of slightly positive and negative impacts on the predictions.
# - For `bmi` values greater than `0.05`, the ALE increases sharply, indicating that higher `bmi` values significantly increase the predicted diabetes progression.
# 
# ## Nonlinear Relationships:
# - The curve's shape suggests a nonlinear relationship between `bmi` and the target:
#   - Initially, the effect of increasing `bmi` is small and fluctuates.
#   - After a certain threshold (`bmi > 0.05`), the effect becomes strongly positive and grows sharply.
# 
# ## Feature Distribution:
# - The vertical ticks show that most data points fall between `-0.05` and `0.1`, meaning the model has more reliable estimates in this range.
# - Sparse ticks in extreme values (e.g., `bmi < -0.1` or `bmi > 0.15`) suggest these regions may have higher uncertainty due to less data.
# 
# ---
# 
# # Takeaways
# 
# ## Low `bmi` Values:
# - Tend to reduce predicted diabetes progression compared to the baseline.
# 
# ## Moderate `bmi` Values:
# - Have minimal impact on predictions, fluctuating slightly around the baseline.
# 
# ## High `bmi` Values:
# - Significantly increase predicted diabetes progression, indicating that `bmi` is an important predictor in the model for individuals with high BMI.
# 
# ---
# 
# # Using This Insight
# 
# ## Feature Importance:
# - The sharp increase at higher `bmi` values highlights this feature's strong influence on predictions, especially in the upper range.
# 
# ## Model Trust:
# - The dense tick marks in the middle range suggest the model is more trustworthy here due to better data coverage.
# - Sparse regions may require additional data or caution in interpretation.
# 
# ## Target Interventions:
# - For applications like healthcare, this insight shows that higher BMI is strongly associated with increased diabetes progression, suggesting interventions might focus on reducing BMI for at-risk individuals.

# %%
# NOTE: notebook magic commented for local script use: ! pip install interpret

# %%
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

# Step 1: Load Diabetes Dataset
diabetes_data = load_diabetes()
X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = diabetes_data.target
feature_names = X.columns

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Explainable Boosting Machine (EBM)
# EBM is a GAM-based algorithm from the `interpret` package
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = ebm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (EBM): {mse:.2f}")

# Step 5: Global Explanation
# The global explanation shows how each feature contributes to predictions
global_explanation = ebm.explain_global()

# Show the global explanation in an interactive visualization
show(global_explanation)

# Step 6: Local Explanation for a Specific Instance
# Choose a specific instance to explain
i = 10
local_explanation = ebm.explain_local(X_test.iloc[i:i+1], y_test[i:i+1])

# Show the local explanation in an interactive visualization
show(local_explanation)

# %%
pass
