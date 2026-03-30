# Lecture 10 Notes: Explainability and Interpretability

> Lecture number: 10
> Lecture slug: `lecture_10_explainability_interpretability`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What This Lecture Is About

This lecture is about one of the central questions in modern machine learning: not only how to make good predictions, but also how to understand them well enough to trust, debug, audit, and communicate them.

As models become more complex, prediction quality often improves, but transparency usually decreases. A linear model or a small decision tree can often be inspected directly. A Random Forest, Gradient Boosting model, or neural network usually cannot. This creates a practical problem: the model may be accurate, but people still need to know why it behaves the way it does.

That is why explainability and interpretability matter. They connect model behavior to human reasoning.

## 2. Where XAI Sits in the ML Workflow

In the CRISP-DM pipeline, explainability belongs mainly to the modeling and evaluation phases.

This is important because explainability is not a decorative add-on after training. It is part of model validation:

- to check whether the model relies on reasonable signals,
- to detect leakage, bias, spurious correlations, or overfitting,
- to communicate results to business or domain experts,
- to decide whether a model is safe enough to deploy.

In practice, explainability often feeds back into earlier stages:

- back to data understanding if suspicious patterns are found,
- back to feature engineering if important signals are missing,
- back to model selection if a simpler but more transparent model is preferable.

## 3. Interpretability vs. Explainability

The lecture makes a useful distinction between these two terms.

### Interpretability

Interpretability means understanding how the model works internally. The internal logic is visible enough that a human can trace how inputs affect outputs.

Examples:

- linear regression,
- logistic regression,
- small decision trees,
- rule-based systems.

If a decision tree says:

- if income > 50,000 and credit score > 700, approve,

then the reasoning process is directly visible. This is interpretability.

### Explainability

Explainability means describing model behavior or individual predictions in a way that humans can understand, even if the underlying model is complex.

Examples:

- explaining why a Random Forest approved a loan,
- showing that high income and good credit score pushed the prediction upward,
- showing that debt ratio pushed the prediction downward.

In other words:

- interpretability is about understanding the mechanism,
- explainability is about understanding or communicating the decision.

These two ideas overlap, but they are not identical. A model may be hard to interpret internally yet still be explainable through post-hoc tools.

## 4. Why This Matters

The lecture emphasizes five practical reasons.

### 1. Trust and transparency

People are more likely to trust a model if they understand what drives its predictions. This is especially important in:

- healthcare,
- finance,
- hiring,
- public sector decision systems,
- fraud detection.

If a doctor sees that a prediction is driven by clinically meaningful variables, trust increases. If the model behaves strangely and nobody can explain it, trust collapses.

### 2. Accountability and ethics

Explainability helps reveal whether the model uses unfair or problematic patterns. It does not automatically fix bias, but it helps expose it.

For example, a model may appear accurate but rely on a proxy for sensitive information. Without interpretability tools, that problem may stay hidden.

### 3. Regulatory compliance

In many applied settings, organizations must justify model-driven decisions. Even if the legal requirement is not phrased as “full explanation of the entire model,” there is often a real operational need to provide understandable reasons for decisions.

### 4. Debugging and model improvement

Explainability is a strong debugging tool. If a housing model overweights a weak feature and underweights location, or if a fraud model pays too much attention to an artifact of data collection, explanation tools can reveal it.

### 5. Actionable insights

A useful explanation can do more than justify a prediction. It can support a decision:

- why a customer was denied a loan,
- what changes would improve creditworthiness,
- what factors drove a risk score,
- what signals separate high-value and low-value customers.

## 5. When to Use Interpretability and When to Use Explainability

The lecture frames this partly as an audience question.

### Use interpretability when:

- you are debugging the model,
- you need auditability,
- you need tight control over the model’s internal logic,
- you operate in a high-risk setting and simple models are acceptable,
- you want a model whose reasoning is inherently visible.

Typical audience:

- data scientists,
- ML engineers,
- validators,
- auditors,
- regulators.

### Use explainability when:

- the model is complex,
- stakeholders are not technical,
- you need case-by-case reasoning,
- the main goal is communication and trust.

Typical audience:

- managers,
- customers,
- doctors,
- analysts,
- business owners,
- investigators.

This distinction matters because the best explanation tool depends on who needs the answer and what kind of answer they need.

## 6. Global vs. Local Interpretability

This is one of the most important distinctions in the lecture.

### Global interpretability

Global interpretability asks:

- What does the model generally learn over the full dataset?
- Which features matter overall?
- What is the average effect of a feature?
- What patterns dominate model behavior?

Examples:

- overall feature importance,
- PDPs,
- aggregated SHAP summary plots,
- coefficient tables in linear models.

Global explanations are useful when you want to understand the model as a whole.

### Local interpretability

Local interpretability asks:

- Why did the model make this specific prediction for this specific case?

Examples:

- SHAP force plot for one instance,
- LIME explanation for one data point,
- path through a decision tree for one record.

Local explanations are useful when a single decision must be justified.

### Important limitation

Local and global explanations solve different problems. A model may have plausible local explanations and still be globally problematic. Conversely, a globally sensible model may make some odd local decisions.

Students should not confuse:

- “I explained one prediction”

with:

- “I understand the whole model.”

## 7. Intrinsic vs. Post-hoc Interpretability

### Intrinsic interpretability

An intrinsically interpretable model is understandable by design.

Examples:

- linear regression,
- logistic regression,
- decision trees,
- rule lists,
- simple scoring models.

These models are preferred when transparency is part of the requirement.

### Post-hoc interpretability

Post-hoc methods are applied after model training to explain a model that is not transparent by nature.

Examples:

- SHAP,
- LIME,
- Permutation Importance,
- PDP,
- ALE.

These methods are necessary when using complex models such as:

- Random Forest,
- Gradient Boosting,
- XGBoost,
- neural networks.

### Core trade-off

The lecture repeatedly points to a central trade-off:

- more complex models may achieve better predictive performance,
- simpler models are often easier to inspect and justify.

This is not a universal rule, but it is a recurring engineering trade-off. In regulated or high-stakes settings, interpretability may matter as much as raw accuracy.

## 8. Interpretable Classical Models

The lecture uses classical models as the baseline for understanding interpretability.

### Linear regression

Linear regression predicts:

\[
\hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
\]

Interpretation:

- each coefficient represents the change in predicted target for a one-unit increase in the feature,
- all other features are held constant.

Important points:

- positive coefficient means the target increases as the feature increases,
- negative coefficient means the target decreases as the feature increases,
- larger absolute value suggests stronger effect, but only if features are on comparable scales.

This last caveat matters. Raw coefficient magnitude is not always comparable if features use different units. Standardization is often needed if you want a fair magnitude comparison.

The slides also mention:

- p-values,
- confidence intervals,
- overall model statistics such as R-squared and F-statistic.

These are important because interpretability is not only “what sign does the coefficient have?” It also includes:

- whether the effect is statistically supported,
- how uncertain that estimate is,
- whether the overall model fit is meaningful.

### Logistic regression

Logistic regression models the log-odds of class membership:

\[
\log \frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
\]

Interpretation:

- positive coefficient increases log-odds of the positive class,
- negative coefficient decreases log-odds,
- exponentiating a coefficient gives an odds ratio.

This is already slightly harder to interpret than linear regression because the model is linear in log-odds, not in probability. A one-unit change in a feature does not correspond to a fixed probability change everywhere. The effect on probability depends on the region of the sigmoid curve.

Still, logistic regression remains relatively interpretable, especially compared with ensemble models.

### Decision trees

Decision trees are inherently interpretable because they express decisions as a sequence of splits.

Strengths:

- logic is visible,
- rules are easy to communicate,
- nonlinear relationships can be captured,
- interactions may appear naturally through branches.

Limitations:

- large trees become hard to read,
- small changes in data can change the tree,
- a readable tree is often less accurate than a strong ensemble.

The notebook practice begins with exactly this kind of intuitive tree-based reasoning on a credit-style dataset using features such as income and credit score.

## 9. Feature Importance in Tree Models and Ensembles

The lecture explains feature importance in decision trees through impurity reduction.

### Impurity-based importance

At a split, the tree chooses a feature that reduces impurity:

- Gini impurity,
- entropy,
- or another split criterion depending on the model.

A feature receives importance based on how much impurity reduction it contributes across all nodes where it is used.

In ensembles such as Random Forest:

- this contribution is aggregated across all trees.

This is often called Mean Decrease in Impurity, or MDI.

### Why it is useful

- quick global overview,
- built into many tree-based models,
- easy to visualize as a ranked bar chart.

### Why it must be interpreted carefully

Impurity-based importance can be biased:

- toward features with many possible split points,
- toward high-cardinality continuous or categorical variables,
- and it may distribute importance in unstable ways when features are correlated.

So feature importance is useful, but it is not a complete explanation.

## 10. Permutation Importance

Permutation importance is one of the cleanest model-agnostic methods in the lecture.

### Core idea

1. Measure the model’s baseline performance.
2. Shuffle one feature.
3. Recompute performance.
4. Measure how much performance drops.

If shuffling a feature hurts performance a lot, the model depends strongly on that feature.

### Why it is attractive

- model-agnostic,
- simple to explain,
- directly linked to predictive performance,
- works for many models.

### Important limitations

- if features are strongly correlated, shuffling one feature may not cause a large drop because other correlated features can substitute for it,
- results depend on the evaluation metric,
- unstable models may give unstable importance values,
- importance does not reveal direction of effect, only reliance.

So permutation importance answers:

- “How much does the model rely on this feature?”

but not:

- “Does a higher value increase or decrease the prediction?”

In the practice notebook, permutation importance is shown first on synthetic data and then on the Wine dataset, which is useful because it demonstrates both the mechanics and a real dataset example.

## 11. Partial Dependence Plots (PDP)

PDP is introduced as a global interpretation method.

### What PDP shows

For one feature, PDP estimates the average model prediction as that feature changes while other features are held fixed through averaging over the dataset.

For two features, PDP shows their joint average effect.

### The logic

For a chosen feature value:

1. replace that feature in all rows with the chosen value,
2. keep the remaining features as they are,
3. make predictions,
4. average them.

Repeat over many values to obtain the dependence curve.

In compact notation, for a selected feature subset \(S\), the partial dependence function is estimating something like

\[
\hat{f}_S(x_S) = \mathbb{E}_{X_C}[f(x_S, X_C)]
\]

where \(X_C\) denotes the remaining features that are averaged out. This is useful because it clarifies exactly what PDP is doing: averaging model responses over the marginal distribution of the other variables.

### What students should understand

PDP shows average marginal effect, not individual behavior.

That means:

- it is global rather than local,
- it smooths over heterogeneity,
- it can hide subgroup effects.

### Very important assumption

The slides explicitly warn that PDP works best when the relevant features are not strongly correlated.

Why?

Because replacing one feature while keeping correlated features unchanged may create unrealistic data points. The model is then evaluated on combinations that barely exist in the real data.

This is the key conceptual weakness of PDP.

In the bike sharing example, PDP is used to show how variables such as temperature and humidity affect predictions. That is a good practical example because it makes the 1-way and 2-way interpretation visible.

## 12. Accumulated Local Effects (ALE)

ALE is presented as a more reliable alternative to PDP when correlation or interactions matter.

### Why ALE is needed

PDP can be misleading when correlated features produce unrealistic synthetic combinations. ALE reduces this problem by measuring local changes in prediction inside small intervals where the data actually exists.

### How ALE works conceptually

1. Split the feature range into bins.
2. For each bin, slightly change the feature within that local interval.
3. Measure how predictions change.
4. Accumulate these local effects across bins.

This gives a function showing how the feature influences the prediction over its range.

### Why ALE is often better than PDP

- it is more robust with correlated features,
- it focuses on local changes rather than unrealistic global substitutions,
- it is often more trustworthy for nonlinear models.

### Limitation

ALE is usually less immediately intuitive to beginners than PDP. It is conceptually cleaner, but visually and mathematically slightly harder to grasp.

In the practice notebook, ALE is demonstrated with `alibi` on the Iris dataset using logistic regression. This is useful because students can compare the idea of local accumulated effect with the more naive global averaging used in PDP.

## 13. SHAP

SHAP is one of the most important techniques in the lecture.

### Core idea

SHAP, or SHapley Additive exPlanations, is based on cooperative game theory.

Each feature is treated as a “player” in a game, and the prediction is the “payout.” SHAP tries to assign a fair contribution to each feature.

The general additive explanation form is:

\[
f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j
\]

where:

- \(\phi_0\) is the base value,
- \(\phi_j\) is the contribution of feature \(j\) for that specific prediction.

This decomposition is elegant, but students should keep two cautions in mind:

- SHAP explains the model output relative to a baseline expectation, not real-world causality,
- when features are dependent, attribution becomes more subtle because correlated variables can share or redistribute credit in non-obvious ways.

### Why SHAP is powerful

It gives both:

- global insight when aggregated over many instances,
- local explanations for individual predictions.

### SHAP summary plot

The summary plot is a global view:

- features are ranked by importance,
- x-axis shows SHAP value,
- color shows whether the feature value is high or low,
- spread shows variability of feature impact.

This is not just a feature ranking plot. It also shows direction and heterogeneity:

- whether high values push predictions up or down,
- whether the effect is stable or varies across observations.

That spread is often the most informative part of the plot. If the same feature shows both strong positive and strong negative SHAP values across different observations, that usually signals nonlinearity, interactions, or subgroup-specific behavior.

### SHAP force plot

The force plot is local:

- starts from a base value,
- shows features pushing the prediction higher or lower,
- ends at the final output for one instance.

This is one of the clearest local explanation formats when it renders well.

### Limitations

- exact SHAP can be computationally expensive,
- explanations can still be misleading if the model is trained on biased or leaky data,
- feature dependence complicates interpretation,
- a mathematically elegant attribution is not the same as causal explanation.

In the practical notebooks, SHAP is demonstrated on tree-based regression tasks such as California Housing and Diabetes-style regression examples. That is a good fit because SHAP is especially strong for tree models.

## 14. LIME

LIME stands for Local Interpretable Model-agnostic Explanations.

### Core idea

For one selected instance:

1. generate perturbed samples near that instance,
2. obtain predictions from the original complex model,
3. fit a simple interpretable surrogate model locally,
4. use the surrogate to explain the local prediction.

So LIME does not claim to explain the full model. It explains how the model behaves in a small neighborhood around one point.

### Why LIME is useful

- easy to understand conceptually,
- local explanations are often intuitive,
- model-agnostic,
- useful for case-by-case inspection.

### Key limitation

The explanation depends on:

- how the neighborhood is sampled,
- how locality is weighted,
- what surrogate model is fitted.

This means LIME can be unstable. Two nearby perturbation setups can produce somewhat different explanations. Students should understand that LIME is a useful approximation, not an exact decomposition of model behavior.

That is why LIME is best used as a local diagnostic tool for a particular case, not as a globally consistent attribution system.

In the notebooks, LIME is used to explain individual predictions of complex regressors, which is exactly the kind of local use case for which it was designed.

## 15. Explainability in Ensemble Methods

The lecture spends time on ensembles because they are common and powerful but much harder to understand directly.

### Why ensembles are harder to interpret

Ensemble models combine many weak or base learners:

- Random Forest averages many trees,
- Gradient Boosting builds trees sequentially,
- Stacking combines models through a meta-model.

This improves predictive performance, but it makes internal logic much less visible.

### Ensemble-specific explanation options

The lecture lists both model-specific and model-agnostic approaches.

#### Model-specific

- impurity-based feature importance,
- mean decrease in accuracy,
- loss reduction contribution in boosting models.

#### Model-agnostic

- permutation feature importance,
- PDP,
- ALE,
- SHAP,
- LIME.

This is an important pattern for students:

- model-specific methods are often faster and more tightly connected to the model structure,
- model-agnostic methods are more flexible and portable across algorithms.

## 16. Limitations and Considerations

This part of the lecture is important because students often over-trust explanation tools.

### 1. Explanations do not guarantee correctness

A model can have a clean explanation and still be wrong.

Explanation tools show how the model behaves, not whether the model is valid or fair in a broader sense.

### 2. Correlation makes interpretation harder

Correlated features create attribution ambiguity:

- which feature is truly responsible?
- which feature is just a proxy?

This affects:

- PDP,
- permutation importance,
- SHAP interpretation,
- impurity-based tree importance.

### 3. Local explanations are not global understanding

LIME or SHAP on one example does not explain the entire model.

### 4. Global explanations can hide subgroups

Average plots may conceal that the model behaves very differently for different segments of the data.

### 5. Explanations are not causal

This is one of the most important conceptual warnings.

If SHAP says a feature contributed positively, that does not mean changing the feature in the real world will cause the same outcome change. The explanation is about the model’s learned predictive relationship, not necessarily a real causal mechanism.

### 6. Cost and scalability

Methods such as SHAP and LIME may become expensive on large datasets or complex pipelines.

### 7. Ethics and compliance

Explanations may expose unfairness, but they do not solve unfairness automatically. A biased dataset can still produce well-explained but unfair predictions.

## 17. Tools and Libraries from the Lecture

The lecture references a number of practical libraries.

### SHAP

Strong choice for:

- tree-based models,
- global + local explanation,
- summary and force plots.

### LIME

Useful for:

- local case-specific explanation,
- quick model-agnostic inspection.

### InterpretML

Useful for:

- glassbox models,
- Explainable Boosting Machines,
- interactive interpretability workflows.

### ELI5

Useful for:

- quick debugging and feature contribution views,
- model inspection in scikit-learn style workflows.

### PiML

Useful for:

- integrated interpretable ML workflow,
- model development and explanation in a structured framework,
- regulated or validation-heavy environments.

### Scikit-learn built-ins

Useful for:

- permutation importance,
- PDP via `PartialDependenceDisplay`,
- feature importance in tree models.

## 18. Practical Notebook Map

This lecture is especially practical. The notebooks cover several distinct explainability workflows.

### 1. `xAI - lecture.ipynb`

This is the main lecture demo notebook. It contains a sequence of hands-on explainability examples:

- a small decision-tree example with income and credit score,
- permutation importance on synthetic data,
- permutation importance on the Wine dataset,
- PDP on the Bike Sharing dataset,
- ALE with `alibi` on the Iris dataset,
- SHAP on California Housing with XGBoost,
- LIME on California Housing,
- classical model summaries with `statsmodels`.

This notebook is valuable because it mirrors the structure of the slides. It moves from intrinsically interpretable models to increasingly advanced post-hoc explainers.

### 2. `xAI Demo - piml.ipynb`

This notebook shows a workflow built around the `piml` library.

The main stages are:

- data loading,
- data preparation,
- feature selection,
- EDA,
- model training,
- model-specific interpretation,
- model-agnostic explanation,
- diagnostics and comparison.

This notebook is useful because it shows explainability not as one isolated chart, but as part of a broader model development process.

### 3. `xai - demo.ipynb`

This notebook contains focused examples with:

- SHAP summary and dependence plots,
- LIME,
- ALE,
- InterpretML / EBM-style tools.

It is useful as a compact sandbox for comparing explanation methods on a regression problem.

## 19. What Students Should Remember Technically

After this lecture, students should be able to answer the following questions clearly.

### What is the difference between interpretability and explainability?

- Interpretability is direct understanding of model internals.
- Explainability is making model behavior understandable, often for complex models.

### What is the difference between global and local explanations?

- Global explanations describe overall model behavior.
- Local explanations justify one prediction.

### What is the difference between intrinsic and post-hoc methods?

- Intrinsic methods come from transparent models.
- Post-hoc methods are added after training to explain opaque models.

### What does permutation importance tell us?

- How much predictive performance depends on a feature.

### What does PDP tell us?

- The average marginal effect of a feature on predictions.

### Why can PDP fail?

- Because correlated features can create unrealistic synthetic combinations.

### Why is ALE often preferred over PDP?

- Because ALE is more robust under correlation and focuses on local feature changes.

### What does SHAP provide?

- A feature attribution framework that supports both local and global analysis.

### What is LIME best for?

- Local approximation of complex model behavior for a single case.

## 20. Key Takeaways

- Explainability is part of model validation, not just presentation.
- Simpler models are easier to interpret, but may be less powerful.
- Complex models often need post-hoc explanation tools.
- No explanation method is universally correct; each answers a different question.
- Correlation, leakage, sparsity, instability, and bias can distort explanations.
- Explanations are descriptive of model behavior, not proofs of causality.

## 21. Quick Revision Questions

1. Why is a coefficient table interpretable, but SHAP usually considered post-hoc?
2. Why can permutation importance underestimate a feature’s value when strong correlation exists?
3. Why is PDP risky when features are highly dependent?
4. What kind of question is better answered by LIME than by a SHAP summary plot?
5. Why is “this feature has a high SHAP value” not the same as “this feature causes the outcome”?
