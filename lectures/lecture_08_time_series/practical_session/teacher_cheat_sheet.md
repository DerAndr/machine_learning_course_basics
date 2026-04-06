# Teacher Cheat Sheet

## Session Goal

Run one practical session on time series forecasting around a more interesting real-world demand problem: forecasting **daily bike rentals in Seoul** from the OpenML dataset `seoul_bike_sharing_demand` (`id=46297`).

Both groups start from the same daily series, the same weekly seasonal baseline, and the same diagnostics. They then split into two modelling tracks:

- Group A: ARIMAX-style non-seasonal model and SARIMAX
- Group B: Random Forest and CatBoost

Prophet is used as a short whole-class comparison model at the end.
The notebook also ends with two optional teaching blocks:

- a minimal `tsfresh` feature-extraction example
- one controlled comparison of the same Random Forest with and without `tsfresh` features

Important modelling assumption:

- for the stronger models, assume a short-range **weather forecast is available** for the 28-day prediction window
- this allows the notebook to use temperature, humidity, rainfall, and snowfall as exogenous or tabular features

## Recommended Timing

1. `0-10 min`: shared setup, explain the difference between baseline, classical forecasting, feature-based forecasting, and Prophet
2. `10-25 min`: teacher-led walkthrough of the shared diagnostics section
3. `25-60 min`: group work inside the student notebook
4. `60-75 min`: groups finish runs and prepare one short summary
5. `75-90 min`: whole-class debrief, optional `tsfresh` and interpretation discussion

## Reference Outcome From The Teacher Notebook

These values are not the teaching goal, but they are useful as a quick sanity check on the current notebook version:

- Seasonal naive baseline: RMSE about `9245`
- ARIMAX(3,0,2) with weather/calendar regressors: RMSE about `2300`
- SARIMAX(1,0,1)x(1,0,1,7) with weather/calendar regressors: RMSE about `2187`
- Random Forest with rich lag/weather/calendar features: RMSE about `3276`
- CatBoost on `log(target)` with rich lag/weather/calendar features: RMSE about `2021`
- Prophet with weather/calendar regressors and monthly seasonality: RMSE about `2710`
- Random Forest with manual rich features plus `tsfresh`: RMSE about `3595`

Important teaching note:

- in the current notebook version, `tsfresh` does **not** improve the Random Forest holdout result
- keep this result, because it is pedagogically useful: more automatic features do not guarantee better forecasting quality

If a group is dramatically worse than the weekly seasonal baseline, inspect forecast alignment, recursive logic, and whether it accidentally used future information.

If a group is dramatically better than the seasonal baseline, check whether it is correctly using only the exogenous values that are assumed known in advance.

## What Each Group Should Do

### Group A: ARIMAX-Style Model And SARIMAX

Focus cells:

- shared diagnostics
- TimeSeries CV comparison
- ARIMAX-style fit and forecast
- SARIMAX fit and forecast
- final comparison table

What they should notice:

- the daily demand series has strong weekly structure
- the ACF/PACF plots support a weekly seasonal hypothesis
- ordinary random `KFold` is not a valid validation strategy for this problem
- a non-seasonal ARIMA core becomes much stronger once exogenous weather and calendar inputs are added
- adding explicit weekly seasonality on top of that still helps
- the strongest classical model here is the season-aware, regressor-aware one

Expected talking point:

- "The non-seasonal ARIMA-style model was already strong once it saw exogenous signals, but SARIMAX improved further by encoding the weekly cycle explicitly."

### Group B: Random Forest And CatBoost

Focus cells:

- shared diagnostics
- TimeSeries CV comparison
- lagged supervised dataset construction
- recursive forecast logic
- Random Forest forecast
- CatBoost forecast
- final comparison table

What they should notice:

- a tabular ML model can work on time series only after explicit lag-based feature engineering
- ordinary shuffled cross-validation can look much better than it should because of time leakage
- recursive forecasting is necessary for a real future horizon
- Random Forest is the simpler tree baseline
- CatBoost can improve on Random Forest even with the same lagged table
- the log-target version of CatBoost can stabilize the problem and improve accuracy further

Expected talking point:

- "The machine learning models only became valid forecasting models once we built lag features and predicted recursively instead of leaking the future."

## Prophet Comparison

Use the Prophet section as a short contrast after the group work. It should not replace the main student modelling tasks.

The point is to show what a higher-level additive forecasting API simplifies and what it hides.

## Interpretation Blocks

The teacher notebook includes additional interpretation cells after the main models:

- ARIMAX and SARIMAX: exogenous coefficient tables
- Random Forest and CatBoost: top feature importances
- Prophet: regressor coefficients

Use these blocks carefully:

- do not compare raw coefficient magnitudes directly across different model families
- explain that `is_functioning_day` is almost a regime or shutdown flag, so it can dominate many plots
- if students get stuck on one giant feature, use that to discuss feature meaning rather than treating it as an error

## What To Check Quickly During Class

- Students understand why the weekly seasonal baseline is the first comparison point.
- Students understand why `TimeSeriesSplit` is valid here and shuffled `KFold` is not.
- Group A can explain why the non-seasonal ARIMAX-style model and SARIMAX behave differently on this dataset.
- Group A understands that weather features are only allowed because the notebook assumes a future weather forecast is available.
- Group B is using lagged features and recursive forecasting rather than leaking future values.
- Group B can explain why CatBoost and Random Forest can share the same lagged table.
- Students understand that `tsfresh` is an optional experimentation tool, not an automatic quality upgrade.
- Both groups can articulate one advantage and one drawback of their modelling approach.

## Minimal Deliverable From Each Group

Ask each group to report:

1. which baseline they used and how strong it was
2. which model family they built
3. which evidence from the diagnostics supported their modelling choice
4. whether their model beat the seasonal naive baseline
5. one concrete weakness of their final approach
6. one methodological lesson from the notebook

## Debrief Prompt

Use these final prompts:

- Why is a strong baseline especially important in time series work?
- Why is `TimeSeriesSplit` safer than shuffled `KFold` for this problem?
- What does SARIMAX capture that the non-seasonal ARIMAX-style model misses here?
- Why is recursive forecasting a methodological requirement for Random Forest and CatBoost?
- When would you still prefer a tree model over a classical seasonal model?
- What does Prophet automate well, and what should a practitioner still verify manually?
- Why might an automated feature library increase complexity without improving the final forecast?
