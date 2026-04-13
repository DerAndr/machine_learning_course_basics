# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %%
# NOTE: notebook magic commented for local script use: !pip install piml

# %%
from piml import Experiment
exp = Experiment()

# %%
# Choose BikeSharing
exp.data_loader()

# %%
# Exclude these features one-by-one: "yr", "mnth", "temp"
exp.data_summary()

# %%
# Prepare dataset with default settings
exp.data_prepare()

# %%
exp.feature_select()

# %%
# Exploratory data analysis, check distribution and correlation
exp.eda()

# %%
# Choose model(s), customize if needed, click Run to train, then register the trained models one by one.
exp.model_train()

# %% [markdown]
# # Interpretability and Explainability

# %%
# Model-specific inherent interpretability:  feature importance, main effects and pairwise interactions.
exp.model_interpret()

# %%
# Model-agnostic post-hoc explanability: global methods (PFI, PDP, ALE) and local methods (LIME, SHAP)
exp.model_explain()

# %% [markdown]
# ## Model Diagnostics and Outcome Testing

# %%
exp.model_diagnose()

# %%
exp.model_compare()

# %%
pass
