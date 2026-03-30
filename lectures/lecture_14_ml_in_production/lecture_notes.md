# Lecture 14 Notes: Machine Learning in Production

> Lecture number: 14
> Lecture slug: `lecture_14_ml_in_production`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What This Lecture Is About

This lecture is about what happens after a model looks good in a notebook.

A model that performs well in experimentation is not automatically useful in the real world. To deliver value, it must become part of a production system that is:

- reliable,
- scalable,
- monitorable,
- maintainable,
- aligned with business goals,
- and safe to operate over time.

This is the core idea of ML in production:

- not just training models,
- but building systems around models.

## 2. Why Production ML Matters

The lecture frames production ML as the bridge between research and real-world impact.

In research or offline experimentation, a model may be evaluated on static data under controlled conditions. In production, the situation is very different:

- inputs arrive continuously,
- users behave unpredictably,
- data distributions evolve,
- latency constraints matter,
- failures have operational and business consequences.

The production question is therefore not:

- “Can we train a model?”

but:

- “Can we operate this model as a dependable part of a larger system?”

## 3. Core Challenges in Production ML

The lecture highlights several challenges that are specific to production settings.

### 1. Continuous arrival of new data

Production data is dynamic. New data arrives from:

- logs,
- APIs,
- sensors,
- transactions,
- user interactions.

This means the modeling problem is not static. The data-generating process can evolve.

### 2. Data drift and distribution shifts

A model trained on old data may become less accurate if the production distribution changes.

This can happen because:

- user behavior changes,
- markets change,
- seasonality appears,
- upstream systems change,
- sensors degrade,
- product logic changes.

The lecture repeatedly emphasizes drift because it is one of the main reasons production models fail.

### 3. Versioning of data, code, and models

In production, reproducibility becomes operationally critical.

If a model starts behaving badly, you need to know:

- which code version is running,
- which data was used,
- which feature logic was active,
- which model artifact was deployed,
- which hyperparameters were chosen.

Without versioning, debugging becomes guesswork.

### 4. Latency, scalability, and reliability trade-offs

A production system must satisfy constraints that do not exist in the same way during offline experimentation:

- how fast predictions must be returned,
- how many requests must be handled,
- how much downtime is acceptable,
- how expensive inference can be.

Sometimes the best offline model is not the best production model because it is too slow, too complex, or too hard to maintain.

### 5. Compliance and ethics

Production systems interact with real users and business processes, so they must satisfy legal, privacy, and fairness constraints.

This connects directly back to Lecture 13.

## 4. ML Systems vs. Traditional Software Systems

One of the best conceptual parts of this lecture is the comparison with traditional software.

### Traditional software

Traditional software is usually:

- rule-based,
- deterministic,
- tested against explicit specifications,
- maintained mainly through code updates.

### ML systems

ML systems are usually:

- data-driven,
- probabilistic,
- validated through metrics rather than exact rule conformance,
- continuously affected by changing data,
- maintained through monitoring, retraining, and pipeline updates.

### Why this distinction matters

This changes almost everything about engineering workflow:

- testing,
- deployment,
- maintenance,
- ownership,
- risk management.

A production ML system is not just code plus a model file. It is a coupled system of:

- code,
- data,
- features,
- evaluation logic,
- infrastructure,
- monitoring,
- retraining policy.

## 5. Challenges During Model Development

The slides include a very useful “challenge map” for ML development:

- metric definition,
- baseline model establishment,
- model selection and tuning,
- debugging and testing,
- experiment tracking,
- interpretability and ethics,
- data quality and availability,
- lack of domain knowledge.

This is important because many ML failures are not due to algorithm choice alone.

### Metrics definition

If the wrong metric is optimized, the project may succeed technically and fail practically.

Examples:

- accuracy may be misleading in imbalanced settings,
- MAE and RMSE emphasize different error behavior,
- business value may not align with a generic modeling metric.

### Baseline model

The lecture correctly stresses baseline establishment.

A baseline is essential because it answers:

- are we improving over something simple,
- are we solving the right problem,
- is the extra complexity justified?

### Domain knowledge

Lack of domain knowledge is a production risk. A technically strong model can still be useless if:

- features are misinterpreted,
- constraints are ignored,
- the target does not reflect the real decision process,
- important edge cases are not understood.

## 6. The ML Pipeline

The lecture presents a standard lifecycle:

1. data management and preprocessing,
2. feature engineering,
3. experimentation and model building,
4. validation and testing,
5. deployment,
6. monitoring and feedback loops.

This should not be viewed as a one-time linear sequence. In production, this is a loop.

### Data management and preprocessing

This includes:

- data collection,
- cleaning,
- consistency handling,
- schema management,
- validation.

Weak preprocessing creates weak production systems.

### Feature engineering

Production feature engineering must be:

- reproducible,
- consistent between training and serving,
- documented,
- versioned.

If the feature logic used in training differs from the logic used during inference, the model can degrade immediately.

This failure mode is often called training-serving skew. The model itself may be correct, but the live feature pipeline no longer matches what the model was trained to expect.

### Experimentation and model building

This is the part students already know well:

- trying models,
- tuning hyperparameters,
- comparing metrics.

But production adds a requirement:

- experiments must be traceable.

### Validation and testing

This includes more than offline performance. A production-ready model should be tested for:

- reliability,
- stability,
- fairness,
- robustness,
- inference behavior,
- integration correctness.

### Deployment

Deployment means exposing the model to real usage. That may happen through:

- batch jobs,
- APIs,
- embedded systems,
- event-driven pipelines.

### Monitoring and feedback loops

This is what keeps the model alive after deployment.

A production model without monitoring is not really production-ready.

## 7. Data as the Fuel of ML Projects

The lecture calls data the fuel of ML projects, which is exactly right.

The slides emphasize several data issues:

- availability,
- collection,
- storage and versioning,
- validation,
- preprocessing,
- representation and bias,
- privacy,
- feedback loops,
- inference and scaling.

Students should understand that production ML problems are often data problems first.

A brilliant model cannot compensate for:

- broken data collection,
- schema drift,
- label delays,
- logging bugs,
- biased sampling,
- poor feature definitions.

## 8. Offline vs. Online Evaluation

This distinction is very important in production ML.

## 9. Offline Evaluation

Offline evaluation uses historical data:

- train/test split,
- validation set,
- cross-validation,
- replay datasets.

### Strengths

- fast,
- controlled,
- safe,
- cheap relative to online testing.

### Weaknesses

- may not reflect real-time user behavior,
- may not capture feedback loops,
- may not detect deployment-specific issues,
- may overestimate production usefulness.

Another important limitation is label latency. In many production settings the true outcome arrives much later than the prediction, so even “offline ground truth” may be delayed or incomplete.

## 10. Online Evaluation

Online evaluation uses live or production-like traffic.

The lecture mentions:

- A/B testing,
- multi-armed bandits,
- interleaving,
- deployment strategies with live monitoring.

### Strengths

- reflects current user behavior,
- measures real impact,
- detects mismatch between offline gains and online value.

### Weaknesses

- riskier,
- harder to manage,
- may affect user experience,
- requires stronger guardrails.

Students should remember:

- offline metrics help select candidates,
- online evaluation reveals whether the model actually helps in the real environment.

## 11. CI/CD for ML

The lecture compares ordinary software CI/CD with ML CI/CD, and this is one of the most useful production concepts.

### Traditional CI/CD

Usually focuses on:

- code changes,
- tests,
- build artifacts,
- deployment pipeline.

### ML CI/CD

Must additionally handle:

- data checks,
- model artifacts,
- experiment tracking,
- training pipelines,
- evaluation thresholds,
- retraining flows.

### Why ML CI/CD is harder

Because the system behavior depends on more than code.

A codebase can be unchanged while model behavior changes because:

- training data changed,
- feature pipeline changed,
- upstream data quality changed,
- model artifact changed.

So ML CI/CD must reason about code, data, and artifacts together.

## 12. Training, Evaluation, and Experimentation

The lecture recommends:

- iterative training,
- experiment tracking,
- automated hyperparameter optimization,
- robust evaluation metrics.

This is important because production teams must justify why a specific model was selected.

That means recording:

- datasets used,
- parameter settings,
- evaluation results,
- artifact versions,
- deployment decisions.

Tools mentioned in the lecture include:

- MLflow,
- Weights & Biases.

Students should think of experiment tracking as memory for the team.

## 13. Data Management and Versioning

The lecture mentions tools such as:

- DVC,
- Git-LFS,
- LakeFS.

The main idea is straightforward:

- version datasets and artifacts with the same seriousness as code.

This enables:

- reproducibility,
- rollback,
- auditing,
- debugging,
- comparison across experiments.

Metadata is also important:

- source,
- schema,
- time range,
- sampling logic,
- preprocessing details,
- label generation rules.

## 14. Deployment Strategies

The lecture distinguishes two main inference modes.

### Batch inference

Predictions are generated on a schedule:

- hourly,
- daily,
- weekly,
- or as asynchronous jobs.

Good for:

- lower latency requirements,
- large-scale offline scoring,
- reporting pipelines,
- recommendations prepared ahead of time.

### Real-time inference

Predictions are generated on demand with low latency.

Good for:

- interactive user systems,
- fraud detection,
- live personalization,
- API-based decision support.

### Trade-off

Real-time systems require tighter engineering on:

- latency,
- scalability,
- fault tolerance,
- monitoring.

## 15. Deployment Patterns

The lecture lists several patterns students should recognize.

### Blue-green deployment

Maintain two environments:

- current live version,
- new candidate version.

Traffic can switch cleanly between them, reducing downtime.

This pattern is especially attractive when rollback speed matters, because reverting can be as simple as switching traffic back to the previous environment.

### Canary release

Roll out to a small subset of traffic first.

This is useful because:

- risk is controlled,
- monitoring can catch issues early,
- rollback is easier.

### Feature toggles and dark launches

These patterns let teams:

- separate deployment from full activation,
- expose systems gradually,
- test infrastructure without serving all users.

### A/B testing

Used to compare variants under real traffic and measure business impact.

For ML systems, the winner should rarely be chosen by one metric alone. Accuracy uplift, latency cost, fairness impact, calibration quality, and business KPI movement may point in different directions.

Students should understand that deployment is not only “put the model on a server.” It is also about controlled rollout and risk management.

## 16. Monitoring in Production

Monitoring is one of the most important operational topics in the lecture.

The slides mention monitoring:

- prediction accuracy,
- latency,
- resource usage,
- data drift indicators.

In practice, teams often also monitor intermediate signals such as:

- feature null rates,
- category-frequency changes,
- prediction-score distributions,
- calibration drift,
- slice-level behavior for important user segments.

### Why monitoring matters

A model can silently degrade in production if:

- input distribution changes,
- labels change,
- user behavior shifts,
- the infrastructure slows down,
- a bug appears in preprocessing.

Without monitoring, the team may discover the issue only after business damage has already happened.

This is why production observability must cover both system health and model health. A service can be technically “up” while the model quality has already degraded badly.

## 17. Drift Types

The lecture distinguishes:

- covariate shift,
- label shift,
- concept drift.

### Covariate shift

The distribution of input features \(X\) changes.

Example:

- customer profiles look different from the training period.

### Label shift

The distribution of target labels \(Y\) changes.

Example:

- class prevalence changes over time.

### Concept drift

The mapping \(Y = F(X)\) changes.

This is the hardest case because the relationship the model learned is no longer the same.

Example:

- customer behavior changes after a major product or policy shift,
- fraud patterns evolve,
- demand drivers change.

Students should remember that different drift types require different responses.

## 18. Alerting and Observability

The lecture mentions:

- Prometheus,
- Grafana,
- alert thresholds,
- notifications.

The key engineering idea is that monitoring must lead to action. Metrics alone are not enough.

A strong production setup defines:

- what is measured,
- what thresholds are unacceptable,
- who gets alerted,
- what mitigation action follows.

## 19. Retraining Strategies

The lecture lists three common retraining approaches.

### Scheduled retraining

Retrain at fixed intervals.

Good when:

- drift is expected but not extremely abrupt,
- processes are stable,
- retraining is relatively cheap.

### Trigger-based retraining

Retrain when drift or performance alerts are triggered.

Good when:

- monitoring is mature,
- retraining should be tied to evidence,
- model freshness is important.

### Online or incremental updates

Continuously incorporate new data.

Good when:

- the environment changes rapidly,
- the model or algorithm supports incremental learning,
- the risk of stale models is high.

Students should note that retraining is not always the correct response. Sometimes the problem is:

- broken data,
- changed feature logic,
- label delay,
- infrastructure bugs,
- or a deeper product change.

## 20. Scalability and Infrastructure

The lecture covers both conceptual and platform-level scaling ideas.

### Horizontal scaling

Add more servers or instances.

### Load balancing

Distribute traffic across instances.

### Autoscaling

Adjust resources based on demand.

### Specialized hardware

Use GPUs or TPUs for:

- large-scale training,
- high-throughput inference,
- deep learning workloads.

## 21. Infrastructure Tools

The lecture mentions:

- Kubernetes,
- Kafka,
- serverless functions,
- Ray,
- Dask,
- cloud platforms such as SageMaker, Azure ML, and Vertex AI.

Students do not need to master all these tools immediately. The key idea is:

- production ML requires orchestration, data movement, compute management, and automation.

## 22. Responsible AI Inside Production ML

The lecture explicitly reconnects production with Responsible AI:

- fairness,
- transparency,
- accountability,
- privacy,
- safety.

This is important because production pressure can easily push teams toward short-term performance while ignoring longer-term risk.

A production-ready ML system should therefore be evaluated not only on:

- latency,
- throughput,
- business KPI,

but also on:

- fairness,
- safety,
- privacy compliance,
- monitoring coverage,
- explainability requirements.

## 23. Cost Optimization and ROI

Production ML must justify itself economically.

The lecture emphasizes:

- balancing complexity with performance,
- using infrastructure efficiently,
- measuring business impact,
- prioritizing work that creates value.

This is a good reminder that a technically elegant model may still be the wrong production choice if:

- it costs too much,
- is too slow,
- is too hard to maintain,
- or does not materially improve the business process.

Students should learn to ask:

- what is the measurable value of this model,
- what does it cost to build and operate,
- is the additional complexity worth it?

## 24. Stakeholder Communication and Team Dynamics

Production ML is a team sport.

The lecture stresses:

- setting realistic expectations,
- using non-technical language when needed,
- involving diverse perspectives,
- building collaboration across functions.

This matters because many production ML failures come from misalignment, not only from model weakness.

Examples:

- the technical team optimizes the wrong KPI,
- stakeholders expect impossible certainty,
- domain experts are consulted too late,
- business constraints are discovered only after deployment work starts.

## 25. Keys to Success in ML Projects

The final slides give a practical success framework:

- solve a clearly defined business problem,
- align KPIs with technical metrics,
- validate continuously,
- avoid overengineering,
- use the right tools for the problem,
- empower the team,
- align cross-functionally,
- break work into manageable phases,
- monitor progress and mitigate risk early.

This is a strong production lesson:

- success in ML is rarely just about having the best model,
- it is about solving the right problem with a maintainable system and a coordinated team.

## 26. Practical Notebook Map

The incoming practical notebook for this lecture is lightweight and acts more like a structured production-thinking checklist than a full MLOps build.

### `ML_PRACTICE.ipynb`

This notebook walks through:

- choosing a real dataset,
- doing EDA,
- building preprocessing pipelines,
- trying model selection,
- evaluating results,
- adding explainability,
- comparing manual modeling with AutoML.

Even though it does not implement a full deployment stack, it is useful because it forces students to think in stages:

- data,
- pipeline,
- model,
- evaluation,
- explainability,
- automation.

That progression matches the production lifecycle better than a single train-and-score notebook.

## 27. What Students Should Be Able to Explain After This Lecture

### Why is production ML harder than offline modeling?

Because the system must work continuously under latency, scaling, monitoring, maintenance, and business constraints.

### Why are ML systems different from traditional software systems?

Because behavior depends on data and learned artifacts, not only on deterministic code.

### Why is monitoring necessary after deployment?

Because production data and model behavior can drift, degrade, or fail silently over time.

### What is the difference between offline and online evaluation?

Offline evaluation uses historical data in controlled settings. Online evaluation measures live performance and real impact under actual usage.

### Why are versioning and experiment tracking important?

Because production debugging, rollback, auditing, and reproducibility depend on knowing exactly which data, code, and model were used.

### Why might the best research model not be the best production model?

Because production also optimizes for latency, cost, stability, simplicity, and maintainability.

## 28. Key Takeaways

- Production ML is about systems, not just models.
- Data management, versioning, deployment, monitoring, and retraining are core parts of the ML lifecycle.
- Production evaluation includes both offline and online perspectives.
- Drift detection and feedback loops are necessary for long-term model quality.
- Infrastructure and scaling choices must match workload and business constraints.
- Responsible AI, stakeholder alignment, and ROI are production concerns, not separate topics.

## 29. Quick Revision Questions

1. Why can a model with strong offline metrics still fail in production?
2. What is the difference between covariate shift, label shift, and concept drift?
3. Why does ML CI/CD need more than ordinary code testing?
4. When would batch inference be preferable to real-time inference?
5. Why is stakeholder communication a technical success factor in production ML?
