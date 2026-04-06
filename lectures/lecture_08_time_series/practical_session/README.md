# Time Series Practical Session

This directory contains a 90-minute classroom practical for Lecture 08.

## Files

- `time_series_practical_teacher_90min.ipynb`
- `time_series_practical_student_90min.ipynb`
- `time_series_practical_student_90min.py`
- `teacher_cheat_sheet.md`

## Format

- The teacher and student notebooks preserve the same cell order and overall structure.
- The student notebook contains targeted TODO placeholders in the main modelling cells.
- The student notebook also has a generated Python companion script for easier diffing and review.
- The practical uses the OpenML dataset `seoul_bike_sharing_demand` (`id=46297`) and aggregates the hourly data to daily demand.
- The shared opening section covers:
  - seasonal-naive baseline
  - STL decomposition
  - ACF and PACF
  - a short comparison of `KFold` versus `TimeSeriesSplit`
- The session covers these model families:
  - seasonal naive baseline
  - ARIMA
  - SARIMAX
  - Random Forest
  - CatBoost
  - Prophet
- The final optional section introduces `tsfresh` and asks students to compare one fixed model with and without automatically generated features.
- A practical classroom split is:
  - Group A: ARIMA and SARIMAX
  - Group B: Random Forest and CatBoost
  - Prophet: short whole-class comparison model at the end

## Teaching Intent

- Keep the modelling story simple and concrete on a dataset students can relate to.
- Contrast a classical forecasting workflow with feature-based machine learning and Prophet.
- Use one shared baseline, one shared diagnostics block, and one shared validation story before comparing the models.
- Make the stronger models genuinely competitive by assuming a short-range weather forecast is available for the holdout horizon.
- Be explicit that weather-based regressors are only acceptable under that forecasting assumption.
- Show that automated feature generation is useful for experimentation, but does not guarantee a better holdout result.

## Environment

Run this practical with:

- `uv sync --group time_series --group ensembles`

If you run in Google Colab, install:

- `openml`
- `catboost`
- `prophet`
- `tsfresh`
