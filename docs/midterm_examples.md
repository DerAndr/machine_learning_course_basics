# Machine Learning Midterm: Example Questions

This document provides 5 sample questions to help you understand the format, depth, and mathematical rigor expected on the actual midterm. These exact questions will **not** appear on the exam, but they reflect the core logic of the test.

---

## Guidelines for Writing Formulas During the Exam

The midterm has a strict time limit (150 minutes). Since typing complex math equations can be time-consuming, do not worry about perfect visual formatting. You can submit your mathematical work using any of the following fast methods:

1. **Plain Text Syntax (Recommended):** Use simple text/keyboard operators. You do not need to use LaTeX.
   * *Example:* `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
   * *Example:* `F1 = 2 * (P * R) / (P + R) = 2 * (0.8 * 0.7) / (0.8 + 0.7)` 
2. **Pseudocode / Python Syntax:** If you know Python syntax, use it freely as it is very unambiguous.
   * *Example:* `distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)`
3. **Paper & Photo (Emergency Backup):** If you are running out of time or find typing formulas mechanically difficult, you may quickly write your calculations cleanly on a piece of paper, snap a photo, and send it directly to the instructors via Direct Messages in MS Teams before the official exam time expires. *Please make sure the photo is bright, legible, and directly references the question number.*

*(Note: Please use designated course direct messaging channels for photo backups, not email).*

---

### Question 1: Data Preprocessing (Concept)
You are preprocessing a dataset where a specific numeric column has values mostly ranging from `-10` to `+10`, but contains roughly 5% extreme outliers reaching up to `+1,000,000`. You initially choose `MinMaxScaler`. Another data scientist strongly suggests using `StandardScaler` or `RobustScaler` instead.
**Question:** Explain the specific mathematical risk of using `MinMaxScaler` on this data compared to the other scalers.

> **Expected Answer:**
> `MinMaxScaler` forces the absolute minimum value to `0` and the absolute maximum value to `1`. Because of the extreme `+1,000,000` outliers, 95% of the normal data (between `-10` and `10`) will be squashed mathematically into a microscopic interval extremely close to 0. This destroys the useful variance of the core data for most ML models. `StandardScaler` handles variance much better by centering around the mean, and `RobustScaler` ignores the outliers entirely by using the interquartile range (IQR).

---

### Question 2: Classification Metrics (Calculation)
You have an imbalanced binary classification dataset with exactly 100 samples. 
The True Negative (Class 0) count is 90. The True Positive (Class 1) count is 10.
Your model predicts all 90 negative samples correctly (no False Positives). However, out of the 10 positive samples, it correctly identifies 4, and misses 6.
**Question:**
1. Calculate the **Accuracy** of the model.
2. Calculate the **F1-Score** of the model. Show your intermediate Precision and Recall calculations.

> **Expected Answer:**
> * **TP:** 4
> * **FN:** 6
> * **TN:** 90
> * **FP:** 0
> 
> 1. **Accuracy** = `(TP + TN) / Total` = `(4 + 90) / 100` = **0.94** (or 94%).
> 2. **Precision** = `TP / (TP + FP)` = `4 / (4 + 0)` = **1.0**.
> 3. **Recall** = `TP / (TP + FN)` = `4 / (4 + 6)` = **0.4**.
> 4. **F1-Score** = `2 * (Precision * Recall) / (Precision + Recall)` = `2 * (1.0 * 0.4) / (1.0 + 0.4)` = `0.8 / 1.4` = **0.571**.

---

### Question 3: Ensembles & Interpretability (Concept)
**Question:** Explain why it is generally mathematically trivial to automatically extract Feature Importances (e.g., *Mean Decrease in Impurity*) from a `RandomForestClassifier`, but structurally impossible to do the exact same thing from a hard-voting `VotingClassifier` consisting of a Logistic Regression, an SVM, and a KNN model.

> **Expected Answer:**
> A Random Forest is built entirely out of decision trees. Decision trees natively calculate importance during training by measuring exactly how much every feature split reduces impurity (like Gini or Entropy) internal to the tree structure. A `VotingClassifier` using SVM, KNN, and Logistic Regression contains models that do not "split" features or utilize impurity decreases. Because they all optimize entirely different mathematical objectives (hyperplanes, distance, log-odds), the Voting ensemble lacks a unified internal metric to calculate a global feature importance score.

---

### Question 4: Clustering & Scaling (Concept)
**Question:** Explain why applying K-Means clustering to raw, unscaled features is highly dangerous, whereas passing raw, unscaled features into a Decision Tree Classifier works perfectly fine.

> **Expected Answer:**
> K-Means clustering assigns points to clusters by calculating strict geometric distances (like Euclidean distance). If one feature is measured in thousands (e.g., Salary) and another in single digits (e.g., Age), the "Salary" feature will mathematically dominate the distance calculation simply because its numbers are larger, ruining the clustering. A Decision Tree, however, scans each feature entirely independently to find an optimal horizontal/vertical split threshold. The scale of the feature sequence does not change the optimal splitting point, making trees completely immune to monotonic feature scaling.

---

### Question 5: Regression Error Analysis (Concept/Math Intuition)
You evaluate a regression model and calculate two different error metrics on the exact same test set:
*   **Mean Absolute Error (MAE):** 50
*   **Mean Squared Error (MSE):** 100,000
**Question:** What does this extreme disparity between MAE and MSE explicitly tell you about the distribution of the errors your model is making?

> **Expected Answer:**
> It implies that the model is making reasonably small or moderate errors most of the time (which keeps the absolute, linear MAE relatively low at 50), but it is occasionally making astronomically massive outlier errors. Because MSE squares the prediction mathematically before averaging, those rare giant errors explode the MSE sum heavily compared to the linear MAE. 
