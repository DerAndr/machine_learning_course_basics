# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %%
# Install required libraries (if not already installed)
# NOTE: notebook magic commented for local script use: !pip install statsmodels matplotlib pandas numpy scikit-learn xgboost prophet

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet

# %% [markdown]
# # Data Preparation Function

# %%

def prepare_data(url):
    """
    Loads and prepares data for time series analysis.
    """
    # Load the dataset
    data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    data = data.dropna()

    # Plot the original data
    plt.figure(figsize=(12, 6))
    plt.plot(data['Passengers'], label='Original Data')
    plt.title('Original Data')
    plt.xlabel('Year')
    plt.ylabel('Passengers')
    plt.legend()
    plt.show()

    # Differencing for stationarity
    data['Passengers_Diff'] = data['Passengers'].diff()
    data['Passengers_Diff2'] = data['Passengers'].diff().diff()

    # ACF and PACF plots
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(data['Passengers_Diff2'].dropna(), lags=30, ax=plt.gca(), title="ACF (2nd Differenced Data)")
    plt.subplot(122)
    plot_pacf(data['Passengers_Diff2'].dropna(), lags=30, ax=plt.gca(), title="PACF (2nd Differenced Data)")
    plt.tight_layout()
    plt.show()

    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    return train, test, data

# %% [markdown]
# # Common Function for Model Evaluation and Plotting

# %%
from scipy.stats import probplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import probplot, shapiro
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_and_plot(model_name, actual, predicted, need_visualize=False):
    """
    Evaluates the model and plots:
    - Actual vs Predicted values
    - Q-Q plot of residuals
    - Histogram of residuals
    Includes residual analysis with comments based on results.
    """
    # Calculate residuals
    residuals = actual - predicted

    # Calculate metrics
    rmse = mean_squared_error(actual, predicted, squared=False)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    if need_visualize:
        # Print metrics
        print(f"\n{model_name} Performance Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, label="Actual Data (Test)", color="blue")
        plt.plot(actual.index, predicted, label=f"{model_name} Predictions", color="red", linestyle="--")
        plt.title(f"{model_name}: Actual vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.legend()
        plt.show()

        # Q-Q Plot for residuals
        plt.figure(figsize=(8, 6))
        probplot(residuals, dist="norm", plot=plt)
        plt.title(f"{model_name}: Q-Q Plot of Residuals")
        plt.show()

        # Histogram of residuals
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=20, edgecolor='k', color='skyblue')
        plt.title(f"{model_name}: Histogram of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

        # Residual Analysis
        print("\nResidual Analysis:")

        # Shapiro-Wilk Test for Normality
        stat, p = shapiro(residuals)
        print(f"  Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p:.4f}")
        if p > 0.05:
            print("  Residuals are likely normally distributed (fail to reject H0).")
        else:
            print("  Residuals are not normally distributed (reject H0).")

        # Analyze Q-Q Plot
        print("  Q-Q Plot Analysis:")
        print("    - If residuals align closely with the 45-degree line, they are normally distributed.")
        print("    - Deviations at the tails indicate skewness or heavy tails.")

        # Histogram Analysis
        print("  Histogram Analysis:")
        print("    - The histogram should resemble a bell curve if residuals are normal.")
        print("    - Skewness or multimodal shapes suggest issues with model fit.")

        # ACF Analysis Comment (Optional)
        print("  Recommendation: Check autocorrelation of residuals to ensure independence.")

    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "MSE": mse, "MAPE": mape}

# %% [markdown]
# # SARIMA Model Function

# %%

def sarima_model(train, test, need_visualize=False):
    print("Running SARIMA Model...")
    sarima = SARIMAX(train['Passengers'], order=(1, 2, 1), seasonal_order=(0, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    result = sarima.fit()
    if need_visualize:
        print(result.summary())
        result.plot_diagnostics(figsize=(15, 12))
        plt.show()
    forecast = result.forecast(steps=len(test))

    return evaluate_and_plot("SARIMA", test['Passengers'], forecast, need_visualize)

# %% [markdown]
# # Random Forest Model Function

# %%

def random_forest_model(train, test, need_visualize=False):
    print("Running Random Forest Model...")

    # Prepare lagged features
    train['Lag1'] = train['Passengers'].shift(1)
    test['Lag1'] = test['Passengers'].shift(1)

    # Drop NaNs introduced by shifting
    train = train.dropna()
    test = test.dropna()

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train[['Lag1']], train['Passengers'])

    # Predict on test data
    forecast = rf.predict(test[['Lag1']])

    return evaluate_and_plot("Random Forest", test['Passengers'], forecast, need_visualize)

# %% [markdown]
# # Prophet Model Function

# %%

def prophet_model(train, test, need_visualize=False):
    print("Running Prophet Model...")

    # Prepare data for Prophet
    prophet_train = train.reset_index().rename(columns={"Month": "ds", "Passengers": "y"})
    prophet_test = test.reset_index().rename(columns={"Month": "ds"})

    # Train Prophet
    model = Prophet()
    model.fit(prophet_train)

    # Make predictions
    future = prophet_test[['ds']]
    forecast = model.predict(future)
    predictions = forecast['yhat'].values

    # Plot Prophet Components
    print(f"\nProphet Components:")
    if need_visualize:
        model.plot_components(forecast)
        plt.show()

    return evaluate_and_plot("Prophet", test['Passengers'], predictions, need_visualize)

# %% [markdown]
# # XGBoost Model Function

# %%

def xgboost_model(train, test, need_visualize=False):
    print("Running XGBoost Model...")

    # Prepare lagged features
    train['Lag1'] = train['Passengers'].shift(1)
    test['Lag1'] = test['Passengers'].shift(1)

    # Drop NaNs introduced by shifting
    train = train.dropna()
    test = test.dropna()

    # Train XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(train[['Lag1']], train['Passengers'])

    # Predict on test data
    forecast = xgb.predict(test[['Lag1']])

    return evaluate_and_plot("XGBoost", test['Passengers'], forecast, need_visualize)

# %% [markdown]
# # Main Flow

# %%

dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
train_data, test_data, full_data = prepare_data(dataset_url)

# Initialize Results Table
results = []

# %% [markdown]
# # Run SARIMA

# %%

results.append(sarima_model(train_data, test_data, need_visualize=True))

# %% [markdown]
# # Run Random Forest

# %%

results.append(random_forest_model(train_data, test_data, need_visualize=True))

# %% [markdown]
# # Run Prophet

# %%
results.append(prophet_model(train_data, test_data, need_visualize=True))

# %% [markdown]
# # Run XGBoost

# %%
results.append(xgboost_model(train_data, test_data, need_visualize=True))

# %% [markdown]
# # Create Summary Table

# %%
results_df = pd.DataFrame(results)
print("\nSummary of Model Performance:")
print(results_df)

# %% [markdown]
# # Cross-Validation

# %%
import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given list of data.

    Parameters:
    - data: list or np.array
        Data for which to calculate the confidence interval.
    - confidence: float
        Confidence level (default is 95%).

    Returns:
    - (mean, lower_bound, upper_bound): tuple
        Mean, lower bound, and upper bound of the confidence interval.
    """
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - margin, mean + margin

def cross_validate_models_with_ci(models, data, n_splits=5, confidence=0.95):
    """
    Perform time series cross-validation for multiple models with confidence intervals.

    Parameters:
    - models: dict
        Dictionary of model functions. Example: {"SARIMA": sarima_model, "Prophet": prophet_model}
    - data: pd.DataFrame
        Full dataset with time series data.
    - n_splits: int
        Number of splits for cross-validation.
    - confidence: float
        Confidence level for confidence intervals.

    Returns:
    - cv_results: pd.DataFrame
        DataFrame with average performance metrics and confidence intervals for each model.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []

    for model_name, model_func in models.items():
        print(f"\nCross-validating {model_name}...")
        fold_metrics = {"RMSE": [], "MAE": [], "MSE": [], "MAPE": []}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            print(f"\nFold {fold + 1}/{n_splits}")
            train, test = data.iloc[train_idx], data.iloc[test_idx]

            # Run the model function and evaluate
            metrics = model_func(train, test)
            for key in fold_metrics:
                fold_metrics[key].append(metrics[key])

        # Calculate mean and confidence intervals for each metric
        model_summary = {"Model": model_name}
        for metric in fold_metrics:
            mean, lower, upper = calculate_confidence_interval(fold_metrics[metric], confidence=confidence)
            model_summary[f"{metric} Mean"] = mean
            model_summary[f"{metric} CI Lower"] = lower
            model_summary[f"{metric} CI Upper"] = upper

        cv_results.append(model_summary)

    # Return results as a DataFrame
    return pd.DataFrame(cv_results)

"""# Run Enhanced Cross-Validation"""

# Define models
model_functions = {
    "SARIMA": sarima_model,
    "Random Forest": random_forest_model,
    "Prophet": prophet_model,
    "XGBoost": xgboost_model
}

# Perform cross-validation with confidence intervals
cv_results_df = cross_validate_models_with_ci(model_functions, full_data, n_splits=5)

# %%
# Print cross-validation summary
print("\nCross-Validation Results with Confidence Intervals:")
display(cv_results_df)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Plot RMSE, MAE, and MAPE with Confidence Intervals
plt.figure(figsize=(14, 8))

x = np.arange(len(cv_results_df["Model"]))  # Bar positions
width = 0.25  # Reduced width to fit all three bars comfortably

# RMSE Plot
plt.bar(x - width, cv_results_df["RMSE Mean"], width, label="RMSE Mean", color="royalblue", alpha=0.8, edgecolor="black")
plt.errorbar(x - width, cv_results_df["RMSE Mean"],
             yerr=[cv_results_df["RMSE Mean"] - cv_results_df["RMSE CI Lower"],
                   cv_results_df["RMSE CI Upper"] - cv_results_df["RMSE Mean"]],
             fmt='o', color='black', capsize=5)

# MAE Plot
plt.bar(x, cv_results_df["MAE Mean"], width, label="MAE Mean", color="salmon", alpha=0.8, edgecolor="black")
plt.errorbar(x, cv_results_df["MAE Mean"],
             yerr=[cv_results_df["MAE Mean"] - cv_results_df["MAE CI Lower"],
                   cv_results_df["MAE CI Upper"] - cv_results_df["MAE Mean"]],
             fmt='o', color='black', capsize=5)


# Labels and Titles
plt.xticks(x, cv_results_df["Model"], fontsize=10)
plt.xlabel("Models", fontsize=12)
plt.ylabel("Error Metrics", fontsize=12)
plt.title("Model Performance with Confidence Intervals", fontsize=14)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# %%
# Plot MAPE with Confidence Intervals
plt.figure(figsize=(10, 6))

x = np.arange(len(cv_results_df["Model"]))  # Bar positions

# MAPE Plot
plt.bar(x, cv_results_df["MAPE Mean"], width=0.5, color="teal", alpha=0.8, edgecolor="black", label="MAPE Mean")
plt.errorbar(x, cv_results_df["MAPE Mean"],
             yerr=[cv_results_df["MAPE Mean"] - cv_results_df["MAPE CI Lower"],
                   cv_results_df["MAPE CI Upper"] - cv_results_df["MAPE Mean"]],
             fmt='o', color='black', capsize=5)

# Labels and Titles
plt.xticks(x, cv_results_df["Model"], fontsize=10)
plt.xlabel("Models", fontsize=12)
plt.ylabel("MAPE (%)", fontsize=12)
plt.title("Mean Absolute Percentage Error (MAPE) with Confidence Intervals", fontsize=14)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# %%
pass
