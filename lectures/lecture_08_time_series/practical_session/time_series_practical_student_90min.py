import warnings

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from IPython.display import display
from catboost import CatBoostRegressor
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


DATASET_ID = 46297
DATASET_URL = 'https://www.openml.org/d/46297'
CALENDAR_COLS = ['is_holiday', 'is_functioning_day', 'is_weekend']
WEATHER_EXOG_COLS = [
    'mean_temperature_c',
    'mean_humidity_pct',
    'total_rainfall_mm',
    'total_snowfall_cm',
]
ARIMAX_EXOG_COLS = CALENDAR_COLS + WEATHER_EXOG_COLS
LAG_CONFIG = {'lags': (1, 2, 3, 7, 14, 21, 28), 'windows': (3, 7, 14, 28)}

COLUMN_MAP = {
    '0': 'date',
    '1': 'hour',
    '2': 'temperature_c',
    '3': 'humidity_pct',
    '4': 'wind_speed_m_s',
    '5': 'visibility_10m',
    '6': 'dew_point_c',
    '7': 'solar_radiation_mj_m2',
    '8': 'rainfall_mm',
    '9': 'snowfall_cm',
    '10': 'season',
    '11': 'holiday',
    '12': 'functioning_day',
}


def load_daily_seoul_bikes(dataset_id=DATASET_ID):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, *_ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')
    X = X.rename(columns=COLUMN_MAP)

    frame = X.copy()
    frame['rented_bike_count'] = y.astype(float)
    frame['date'] = pd.to_datetime(frame['date'])
    frame['is_holiday'] = (frame['holiday'] != 'No Holiday').astype(int)
    frame['is_functioning_day'] = (frame['functioning_day'] == 'Yes').astype(int)

    daily = (
        frame.groupby('date')
        .agg(
            rented_bike_count=('rented_bike_count', 'sum'),
            mean_temperature_c=('temperature_c', 'mean'),
            mean_humidity_pct=('humidity_pct', 'mean'),
            mean_wind_speed_m_s=('wind_speed_m_s', 'mean'),
            mean_visibility_10m=('visibility_10m', 'mean'),
            mean_dew_point_c=('dew_point_c', 'mean'),
            total_solar_radiation_mj_m2=('solar_radiation_mj_m2', 'sum'),
            total_rainfall_mm=('rainfall_mm', 'sum'),
            total_snowfall_cm=('snowfall_cm', 'sum'),
            is_holiday=('is_holiday', 'max'),
            is_functioning_day=('is_functioning_day', 'min'),
        )
        .sort_index()
    )
    daily.index = pd.DatetimeIndex(daily.index, freq='D')
    daily['is_weekend'] = (daily.index.dayofweek >= 5).astype(int)
    daily['day_of_week'] = daily.index.dayofweek
    daily['month'] = daily.index.month
    daily['day_of_year'] = daily.index.dayofyear
    daily['week_of_year'] = daily.index.isocalendar().week.astype(int)
    daily['dow_sin'] = np.sin(2 * np.pi * daily['day_of_week'] / 7)
    daily['dow_cos'] = np.cos(2 * np.pi * daily['day_of_week'] / 7)
    daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
    daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)
    return daily


def evaluate_forecast(y_true, y_pred, model_name):
    abs_error = (y_true - y_pred).abs()
    return pd.Series(
        {
            'Model': model_name,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
            'WAPE_pct': abs_error.sum() / y_true.abs().sum() * 100,
        }
    )


def plot_forecast(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    train.plot(ax=ax, label='Train')
    test.plot(ax=ax, label='Test', linewidth=2)
    pred.plot(ax=ax, label='Forecast', linestyle='--')
    ax.set_title(title)
    ax.set_ylabel('Daily rented bikes')
    ax.legend()
    plt.show()


def seasonal_naive_forecast(train_series, horizon_index, season_length=7):
    last_period = train_series.iloc[-season_length:]
    repeats = int(np.ceil(len(horizon_index) / season_length))
    values = np.tile(last_period.values, repeats)[: len(horizon_index)]
    return pd.Series(values, index=horizon_index, name='seasonal_naive')


def stationarity_report(series, label):
    adf_result = adfuller(series)
    kpss_result = kpss(series, regression='c', nlags='auto')
    return {
        'Series': label,
        'ADF_pvalue': adf_result[1],
        'KPSS_pvalue': kpss_result[1],
    }


def make_supervised(frame, lags=(1, 2, 3, 7, 14, 21, 28), windows=(3, 7, 14, 28)):
    base_cols = [
        'rented_bike_count', 'is_holiday', 'is_functioning_day', 'is_weekend',
        'day_of_week', 'month', 'day_of_year', 'week_of_year',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'mean_temperature_c', 'mean_humidity_pct', 'mean_wind_speed_m_s',
        'mean_visibility_10m', 'mean_dew_point_c', 'total_solar_radiation_mj_m2',
        'total_rainfall_mm', 'total_snowfall_cm',
    ]
    data = frame[base_cols].copy()
    for lag in lags:
        data[f'lag_{lag}'] = data['rented_bike_count'].shift(lag)
    for window in windows:
        shifted = data['rented_bike_count'].shift(1)
        data[f'rolling_mean_{window}'] = shifted.rolling(window).mean()
        data[f'rolling_std_{window}'] = shifted.rolling(window).std()
        data[f'rolling_min_{window}'] = shifted.rolling(window).min()
        data[f'rolling_max_{window}'] = shifted.rolling(window).max()
    return data.dropna()


def build_feature_row(history_target, known_row, lags=(1, 2, 3, 7, 14, 21, 28), windows=(3, 7, 14, 28)):
    feature_names = [
        'is_holiday', 'is_functioning_day', 'is_weekend', 'day_of_week', 'month',
        'day_of_year', 'week_of_year', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'mean_temperature_c', 'mean_humidity_pct', 'mean_wind_speed_m_s',
        'mean_visibility_10m', 'mean_dew_point_c', 'total_solar_radiation_mj_m2',
        'total_rainfall_mm', 'total_snowfall_cm',
    ]
    row = {name: float(known_row[name]) for name in feature_names}
    row['is_holiday'] = int(known_row['is_holiday'])
    row['is_functioning_day'] = int(known_row['is_functioning_day'])
    row['is_weekend'] = int(known_row['is_weekend'])
    row['day_of_week'] = int(known_row['day_of_week'])
    row['month'] = int(known_row['month'])
    row['day_of_year'] = int(known_row['day_of_year'])
    row['week_of_year'] = int(known_row['week_of_year'])
    for lag in lags:
        row[f'lag_{lag}'] = float(history_target.iloc[-lag])
    for window in windows:
        recent = history_target.iloc[-window:]
        row[f'rolling_mean_{window}'] = float(recent.mean())
        row[f'rolling_std_{window}'] = float(recent.std())
        row[f'rolling_min_{window}'] = float(recent.min())
        row[f'rolling_max_{window}'] = float(recent.max())
    return pd.DataFrame(row, index=[known_row.name])


def recursive_forecast(model, train_frame, test_frame, feature_columns=None, inverse_transform=None, lags=(1, 2, 3, 7, 14, 21, 28), windows=(3, 7, 14, 28)):
    history = train_frame['rented_bike_count'].copy()
    predictions = []
    if feature_columns is None and hasattr(model, 'feature_names_in_'):
        feature_columns = list(model.feature_names_in_)
    for timestamp, known_row in test_frame.iterrows():
        features = build_feature_row(history, known_row, lags=lags, windows=windows)
        if feature_columns is not None:
            features = features[feature_columns]
        raw_pred = float(model.predict(features)[0])
        pred = inverse_transform(raw_pred) if inverse_transform else raw_pred
        predictions.append(pred)
        history.loc[timestamp] = pred
    return pd.Series(predictions, index=test_frame.index, name='recursive_forecast')


def plot_cv_splits(index, splitter, title):
    positions = np.arange(len(index))
    splits = list(splitter.split(positions))
    fig, axes = plt.subplots(len(splits), 1, figsize=(14, 1.8 * len(splits)), sharex=True)
    if len(splits) == 1:
        axes = [axes]
    for fold_id, (ax, (train_idx, test_idx)) in enumerate(zip(axes, splits), start=1):
        ax.scatter(index[train_idx], np.full(len(train_idx), fold_id), s=12, color='tab:blue', label='Train' if fold_id == 1 else None)
        ax.scatter(index[test_idx], np.full(len(test_idx), fold_id), s=18, color='tab:orange', label='Test' if fold_id == 1 else None)
        ax.set_ylabel(f'Fold {fold_id}')
        ax.set_yticks([])
    axes[0].legend(loc='upper left')
    axes[-1].set_xlabel('Date')
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


def evaluate_cv_strategy(cv_frame, splitter, model_factory):
    X = cv_frame.drop(columns='rented_bike_count')
    y = cv_frame['rented_bike_count']
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = model_factory()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                'fold': fold_id,
                'train_start': X.index[train_idx].min(),
                'train_end': X.index[train_idx].max(),
                'test_start': X.index[test_idx].min(),
                'test_end': X.index[test_idx].max(),
                'future_in_train': X.index[train_idx].max() > X.index[test_idx].min(),
                'rmse': mean_squared_error(y_test, pred) ** 0.5,
            }
        )
    return pd.DataFrame(rows)


def build_tsfresh_demo_frame(series, window=14, max_windows=8):
    series = series.reset_index()
    records = []
    start = max(window - 1, len(series) - max_windows)
    for end in range(start, len(series)):
        history = series.iloc[end - window + 1:end + 1].copy()
        history['window_id'] = end
        history['time_step'] = np.arange(len(history))
        records.append(history[['window_id', 'time_step', 'rented_bike_count']])
    return pd.concat(records, ignore_index=True)


TSFRESH_WINDOW = 14


def plot_ranked_series(series, title, top_n=10, sort_by_abs=True, xlabel='Value'):
    series = pd.Series(series).dropna()
    if sort_by_abs:
        top_index = series.abs().sort_values(ascending=False).head(top_n).index
        top = series.loc[top_index].sort_values()
    else:
        top = series.sort_values(ascending=False).head(top_n).sort_values()
    colors = ['tab:blue' if value >= 0 else 'tab:red' for value in top.values]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(top))))
    ax.barh(top.index.astype(str), top.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def make_tsfresh_supervised(series, window=TSFRESH_WINDOW):
    values = series.reset_index()
    records = []
    target_index = []
    for end in range(window, len(values)):
        history = values.iloc[end - window:end].copy()
        history['window_id'] = end
        history['time_step'] = np.arange(window)
        records.append(history[['window_id', 'time_step', 'rented_bike_count']])
        target_index.append(values.loc[end, 'date'])
    rolled = pd.concat(records, ignore_index=True)
    features = extract_features(
        rolled,
        column_id='window_id',
        column_sort='time_step',
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=0,
    )
    features.index = pd.DatetimeIndex(target_index)
    features.index.name = series.index.name
    return features.sort_index()


def build_tsfresh_row(history_target, current_index, window=TSFRESH_WINDOW):
    history = history_target.iloc[-window:]
    rolled = pd.DataFrame(
        {
            'window_id': np.zeros(len(history), dtype=int),
            'time_step': np.arange(len(history)),
            'rented_bike_count': history.values,
        }
    )
    features = extract_features(
        rolled,
        column_id='window_id',
        column_sort='time_step',
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=0,
    )
    features.index = pd.DatetimeIndex([current_index])
    return features


def recursive_forecast_with_tsfresh(
    model,
    train_frame,
    test_frame,
    base_feature_columns,
    tsfresh_feature_columns,
    inverse_transform=None,
    lags=(1, 2, 3, 7, 14, 21, 28),
    windows=(3, 7, 14, 28),
    tsfresh_window=TSFRESH_WINDOW,
):
    history = train_frame['rented_bike_count'].copy()
    predictions = []
    ordered_columns = list(base_feature_columns) + list(tsfresh_feature_columns)
    for timestamp, known_row in test_frame.iterrows():
        base_features = build_feature_row(history, known_row, lags=lags, windows=windows)
        tsfresh_features = build_tsfresh_row(history, timestamp, window=tsfresh_window)
        features = pd.concat([base_features, tsfresh_features], axis=1)[ordered_columns]
        raw_pred = float(model.predict(features)[0])
        pred = inverse_transform(raw_pred) if inverse_transform else raw_pred
        predictions.append(pred)
        history.loc[timestamp] = pred
    return pd.Series(predictions, index=test_frame.index, name='recursive_tsfresh_forecast')


daily = load_daily_seoul_bikes()

print(f'OpenML dataset id: {DATASET_ID}')
print(f'OpenML URL: {DATASET_URL}')
print(f'Daily observations: {len(daily)}')
print(f'Period: {daily.index.min().date()} to {daily.index.max().date()}')
print('Forecasting assumption: the next-28-day weather forecast is available, so weather variables can be used as exogenous inputs.')

daily.head()

test_horizon = 28
train = daily.iloc[:-test_horizon].copy()
test = daily.iloc[-test_horizon:].copy()

baseline_pred = seasonal_naive_forecast(train['rented_bike_count'], test.index, season_length=7)
baseline_metrics = evaluate_forecast(test['rented_bike_count'], baseline_pred, 'Seasonal naive (7-day)')

print(f'Train period: {train.index.min().date()} to {train.index.max().date()}')
print(f'Test period: {test.index.min().date()} to {test.index.max().date()}')
plot_forecast(train['rented_bike_count'], test['rented_bike_count'], baseline_pred, 'Shared baseline: repeat last observed week')

baseline_metrics.to_frame().T

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
daily['rented_bike_count'].plot(ax=axes[0], color='tab:blue')
axes[0].set_title('Daily bike demand across the full year')
axes[0].set_ylabel('Daily rented bikes')

daily['rented_bike_count'].iloc[-70:].plot(ax=axes[1], color='tab:green')
axes[1].set_title('Last 70 days: weekly seasonality is visible')
axes[1].set_ylabel('Daily rented bikes')
plt.tight_layout()
plt.show()

stl_result = STL(train['rented_bike_count'], period=7, robust=True).fit()
stl_result.plot()
plt.show()

train_diff = train['rented_bike_count'].diff().dropna()
stationarity = pd.DataFrame(
    [
        stationarity_report(train['rented_bike_count'], 'train_raw'),
        stationarity_report(train_diff, 'train_first_difference'),
    ]
)
stationarity

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(train_diff, lags=30, ax=axes[0])
plot_pacf(train_diff, lags=30, ax=axes[1], method='ywm')
axes[0].set_title('ACF of differenced training demand')
axes[1].set_title('PACF of differenced training demand')
plt.tight_layout()
plt.show()

cv_frame = make_supervised(train, **LAG_CONFIG)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
tscv = TimeSeriesSplit(n_splits=5)

plot_cv_splits(cv_frame.index, kfold, 'Ordinary KFold on time series features')
plot_cv_splits(cv_frame.index, tscv, 'TimeSeriesSplit on time series features')


def cv_model_factory():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )


kfold_results = evaluate_cv_strategy(cv_frame, kfold, cv_model_factory)
tscv_results = evaluate_cv_strategy(cv_frame, tscv, cv_model_factory)

cv_summary = pd.DataFrame(
    {
        'Strategy': ['KFold(shuffle=True)', 'TimeSeriesSplit'],
        'Mean_RMSE': [kfold_results['rmse'].mean(), tscv_results['rmse'].mean()],
        'Any_future_in_train': [kfold_results['future_in_train'].any(), tscv_results['future_in_train'].any()],
    }
)

display(cv_summary.round(2))


# TODO: fit a strong non-seasonal ARIMA-style model with exogenous regressors.
# Suggested target setup:
# - order=(3, 0, 2)
# - seasonal_order=(0, 0, 0, 0)
# - exog=train[ARIMAX_EXOG_COLS]
# Then forecast the 28-day holdout and compute arima_metrics.
#
# Optional interpretation step:
# - extract coefficients for ARIMAX_EXOG_COLS from arima_model.params
# - sort them by absolute value
# - discuss why binary operational flags may dominate the weather regressors


# TODO: add the weekly seasonal structure on top of the exogenous classical model.
# Suggested target setup:
# - order=(1, 0, 1)
# - seasonal_order=(1, 0, 1, 7)
# - exog=train[ARIMAX_EXOG_COLS]
# Then forecast the holdout and compute sarimax_metrics.
#
# Optional interpretation step:
# - inspect the coefficients for the exogenous regressors
# - compare them with the ARIMAX-style model
# - ask whether SARIMAX changed the role of weather vs calendar signals


supervised_train = make_supervised(train, **LAG_CONFIG)
feature_cols = supervised_train.drop(columns='rented_bike_count').columns.tolist()
X_train = supervised_train[feature_cols]
y_train = supervised_train['rented_bike_count']

print(f'Supervised training rows: {len(supervised_train)}')
print(f'Number of features: {len(feature_cols)}')
supervised_train.head()


# TODO: fit a stronger Random Forest on the richer lag/weather/calendar table.
# Suggested target setup:
# - n_estimators=1500
# - max_features=0.5
# - max_depth=18
# Then produce recursive forecasts and compute rf_metrics.
#
# Optional interpretation step:
# - build a pd.Series from rf_model.feature_importances_ with index=feature_cols
# - plot the top 10-12 features
# - compare whether the most important features are lags, rolling stats, or exogenous variables


# TODO: fit CatBoost on log(target) using the same rich feature table.
# Suggested target setup:
# - iterations=1000
# - depth=6
# - learning_rate=0.03
# - fit on np.log1p(y_train)
# Then inverse-transform predictions with np.expm1 and compute catboost_metrics.
#
# Optional interpretation step:
# - inspect catboost_model.get_feature_importance()
# - compare the ranking with the Random Forest importance plot
# - discuss why boosting may rely on a slightly different feature mix


# TODO: strengthen Prophet with weather/calendar regressors and one custom monthly seasonality.
# Suggested target setup:
# - add regressors from ARIMAX_EXOG_COLS
# - weekly_seasonality=True
# - changepoint_prior_scale=0.01
# - seasonality_prior_scale=20.0
# - add_seasonality(name='monthly', period=30.5, fourier_order=5)
# Then forecast the holdout and compute prophet_metrics.
#
# Optional interpretation step:
# - use regressor_coefficients(prophet_model)
# - inspect which added regressors have the largest positive or negative coefficients
# - explain why coefficients are on the original target scale


available_metric_names = [
    'baseline_metrics',
    'arima_metrics',
    'sarimax_metrics',
    'rf_metrics',
    'catboost_metrics',
    'prophet_metrics',
]
available_metrics = [globals()[name] for name in available_metric_names if name in globals()]

if available_metrics:
    comparison = (
        pd.DataFrame(available_metrics)
        .sort_values('RMSE')
        .reset_index(drop=True)
    )
    display(comparison.round(2))
else:
    print(
        'Complete at least one forecasting model block above, then build the comparison table here. '
        'The seasonal naive baseline is already available as baseline_metrics.'
    )


tsfresh_demo = build_tsfresh_demo_frame(train['rented_bike_count'], window=14, max_windows=8)

tsfresh_features = extract_features(
    tsfresh_demo,
    column_id='window_id',
    column_sort='time_step',
    default_fc_parameters=MinimalFCParameters(),
    disable_progressbar=True,
    n_jobs=0,
)

tsfresh_features.head()


# TODO: compare one fixed model with and without tsfresh features.
# Suggested experiment:
# - start from the Random Forest section
# - create tsfresh_train_full = make_tsfresh_supervised(train['rented_bike_count'])
# - join tsfresh features to supervised_train
# - fit the same Random Forest on:
#   1. manual rich features
#   2. manual rich + tsfresh features
# - compare RMSE on the holdout
#
# Discussion prompt:
# - did tsfresh add genuinely new signal, or mostly redundant summaries?
# - why can automatic feature generation increase complexity without improving quality?
