# Mini-Project: NYC Airbnb Price Prediction

**Maximum Score:** 30 points  
**Dataset:** New York City, 2019  

**Data Source:**  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)  
*(The Jupyter Notebook includes a code cell that automatically downloads `AB_NYC_2019.csv` via a public mirror so you don't need to authenticate or download it manually).*

---

## Assignment Description

The goal of this mini-project is to build a complete machine learning pipeline to predict Airbnb rental prices in New York City. 
You will tackle real-world "dirty" data, extract useful geographical signals from coordinates, solve the severe right-skewness of the target variable (price), and compare basic linear algorithms against tree-based ensembles.

Please use the provided `airbnb_nyc_student.ipynb` template. You are expected to complete all 5 phases.

---

### Phase 1: Exploratory Data Analysis (EDA)
* Load the dataset and examine the column types. Look for obvious technical artifacts (e.g., apartments with zero price).
* **Missing Values Analysis:** Build a missing values matrix (or use `df.isna().sum()`). Assess the scale of missing data.
* **Target Variable:** Plot the distribution of the `price` column. Assess its shape. How heavily skewed is the data, and does it make sense to log-transform the target?
* **Geography:** Create a simple scatter plot of the coordinates (`latitude` and `longitude`), coloring points by price level or administrative `neighbourhood_group` to visually "draw" the map of NYC.

### Phase 2: Data Preprocessing
* **Handling Missing Values**: Fill (impute) or drop missing values. Briefly justify your decision in the comments (e.g., why did you fill `reviews_per_month` the way you did?).
* **Feature Selection**: Remove technical identifiers (`id`, `host_id`) and raw text fields (`name`, `host_name`) that cannot be used directly in simple models without NLP.
* **Categorical Encoding**: Use an encoder like `OneHotEncoder` or `pd.get_dummies` for categories (neighbourhood group, room type).
* Split the data into `train` and `test` sets (recommended test size: 20%).

### Phase 3: Model Training
* Train **at least 2 different regression models**. 
  * At least one linear model (e.g., `LinearRegression`, `Ridge`, or `Lasso`).
  * At least one tree-based / ensemble model (e.g., `RandomForestRegressor`, `DecisionTreeRegressor`, or `HistGradientBoostingRegressor`).
* *It is highly recommended to use a `Pipeline` to apply transformers cleanly and prevent data leakage.*

### Phase 4: Evaluation and Analysis
* Calculate 3 metrics for each trained model on the test set: **RMSE**, **MAE**, and **R²**.
* Compile the results into a single table/DataFrame for easy comparison.
* Plot the "Feature Importance" for your best tree-based model. Which feature ultimately drives the price the most?

### Phase 5: Business Conclusions
In a separate Markdown cell at the very end of the notebook, briefly answer:
1. Which model performed best and why?
2. Which features or apartment characteristics increase the price in NYC the most?
3. What is the main weakness of the model you built, and what external data would you collect to improve its accuracy?

---

> **Grading Criteria (30 Points Total):**
> * Code structure and proper use of Pandas/Sklearn **[5 points]**
> * Completeness of EDA, charts, and `price` cleaning logic **[5 points]**
> * Robustness of the preprocessing pipeline **[7 points]**
> * Correct model training, validation, and evaluation of regression metrics **[8 points]**
> * Depth of the final business conclusions **[5 points]**
