from __future__ import annotations

from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
LECTURES_ROOT = REPO_ROOT / "lectures"

SUMMARY_OVERRIDES: dict[str, str] = {
    "lecture_01_eda/example_01": (
        "EDA basics: summary statistics, distributions, and pairwise visual analysis."
    ),
    "lecture_01_eda/example_02": (
        "Automated EDA reports and interactive visualization tools for faster inspection."
    ),
    "lecture_02_data_preparation_part_1/example_01": (
        "Missing-value inspection and practical imputation strategies."
    ),
    "lecture_02_data_preparation_part_1/example_02": (
        "Outlier detection, interpretation, and treatment choices."
    ),
    "lecture_02_data_preparation_part_1/example_03": (
        "Categorical encoding methods on toy and tabular data."
    ),
    "lecture_03_data_preparation_part_2/example_01": (
        "Feature selection with filter, wrapper, and embedded methods."
    ),
    "lecture_03_data_preparation_part_2/example_02": (
        "Feature generation patterns and practical feature engineering."
    ),
    "lecture_03_data_preparation_part_2/example_03": (
        "Dimensionality reduction with PCA and manifold-learning methods."
    ),
    "lecture_03_data_preparation_part_2/example_04": (
        "Data splitting, stratification, resampling, and cross-validation setup."
    ),
    "lecture_03_data_preparation_part_2/example_05": (
        "Leakage scenarios and why they invalidate evaluation."
    ),
    "lecture_03_data_preparation_part_2/example_06": (
        "Pipelines for safe preprocessing and modeling workflows."
    ),
    "lecture_04_regression/example_01": (
        "Regression workflow with fitting, diagnostics, and evaluation."
    ),
    "lecture_05_classification_part_1/example_01": (
        "KNN classification mechanics and decision-boundary intuition."
    ),
    "lecture_05_classification_part_1/example_02": (
        "Decision-tree classification and split behavior."
    ),
    "lecture_05_classification_part_1/example_03": (
        "Logistic regression for classification and probability-based decisions."
    ),
    "lecture_06_classification_part_2/example_01": (
        "Imbalanced binary classification with resampling and threshold-aware metrics."
    ),
    "lecture_06_classification_part_2/example_02": (
        "Multiclass classification with native, OvR, and OvO logistic strategies."
    ),
    "lecture_06_classification_part_2/example_03": (
        "Multilabel classification and multilabel metric interpretation."
    ),
    "lecture_06_classification_part_2/example_04": (
        "Ordinal classification with ordered targets and ordinal regression."
    ),
    "lecture_07_ensembles/example_01": (
        "Bagging, boosting, and tree ensembles with tuning and comparison."
    ),
    "lecture_08_time_series/example_01": (
        "Forecasting pipeline with SARIMA, Random Forest, Prophet, and XGBoost."
    ),
    "lecture_09_clustering/example_01": (
        "Clustering workflow with internal metrics, visualization, and multiple algorithms."
    ),
    "lecture_10_explainability_interpretability/example_01": (
        "Model explainability workflow centered on PiML."
    ),
    "lecture_10_explainability_interpretability/example_02": (
        "Interpretability methods with ALE, SHAP, LIME, Alibi, and Interpret."
    ),
    "lecture_11_cross_validation_hpo/example_01": (
        "Cross-validation, pipelines, and hyperparameter search on classification and "
        "regression demos."
    ),
    "lecture_11_cross_validation_hpo/example_02": (
        "Alternative hyperparameter-optimization libraries such as Optuna, Hyperopt, and "
        "scikit-optimize."
    ),
    "lecture_12_intro_neural_networks/example_01": (
        "Neural-network introduction on MNIST with PyTorch."
    ),
    "lecture_13_responsible_ai/example_01": (
        "Guided practice for fairness, bias inspection, and responsible-AI checks."
    ),
    "lecture_14_ml_in_production/example_01": (
        "Guided practice for packaging, validation, and monitoring in production ML."
    ),
}

OPTIONAL_NOTES: dict[str, str] = {
    "lecture_07_ensembles/example_01": (
        "Optional setup note: this example uses CatBoost, LightGBM, and XGBoost. "
        "The baseline environment does not install them by default. "
        "Install the lecture-specific extras with `uv sync --group ensembles`."
    ),
    "lecture_08_time_series/example_01": (
        "Optional setup note: this example uses Prophet and XGBoost. "
        "The baseline environment does not install it by default. "
        "Install the lecture-specific extras with `uv sync --group time_series`."
    ),
    "lecture_10_explainability_interpretability/example_01": (
        "Optional setup note: this example uses PiML. "
        "Install the lecture-specific extras with `uv sync --group xai_piml` if you are using "
        "a compatible Python version. The default Python 3.12 environment does not install PiML "
        "because compatible wheels are not available."
    ),
    "lecture_12_intro_neural_networks/example_01": (
        "Optional setup note: this example uses PyTorch. "
        "The baseline environment does not install it by default. "
        "Install the lecture-specific extras with `uv sync --group neural_networks`."
    ),
    "lecture_11_cross_validation_hpo/example_01": (
        "Optional setup note: this example uses H2O AutoML. "
        "The baseline environment does not install it by default. "
        "Install the lecture-specific extras with `uv sync --group hpo_automl`."
    ),
    "lecture_11_cross_validation_hpo/example_02": (
        "Optional setup note: this example uses Optuna, Hyperopt, and scikit-optimize. "
        "The baseline environment does not install them by default. "
        "Install the lecture-specific extras with `uv sync --group hpo_automl`."
    ),
}


def iter_lecture_dirs() -> list[Path]:
    return sorted(LECTURES_ROOT.glob("lecture_*"))


def extract_title(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
    return nb_path.stem.replace("_", " ").title()


def comment_markdown_line(line: str) -> str:
    return "# " if not line else f"# {line}"


def sanitize_code_line(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith("!") or stripped.startswith("%"):
        return f"{indent}# NOTE: notebook magic commented for local script use: {stripped}"
    if "google.colab" in stripped:
        return f"{indent}# NOTE: Colab-only import commented for local script use: {stripped}"
    return line


def notebook_to_percent_script(nb_path: Path, py_path: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    lines: list[str] = [
        "# /// script",
        f"# source-notebook = \"{nb_path.name}\"",
        "# generated-by = \"tools/sync_lecture_examples.py\"",
        "# ///",
        "",
    ]

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            lines.append("# %% [markdown]")
            for line in cell.source.splitlines():
                lines.append(comment_markdown_line(line))
            lines.append("")
            continue

        if cell.cell_type == "code":
            lines.append("# %%")
            if cell.source.strip():
                for line in cell.source.splitlines():
                    lines.append(sanitize_code_line(line))
            else:
                lines.append("pass")
            lines.append("")

    py_path.write_text("\n".join(lines).rstrip() + "\n")


def build_examples_readme(lecture_dir: Path) -> str:
    readme_title = (
        (lecture_dir / "README.md")
        .read_text()
        .splitlines()[0]
        .removeprefix("# ")
        .strip()
    )
    examples_dir = lecture_dir / "lecture_examples"
    notebook_paths = sorted(examples_dir.glob("example_*.ipynb"))

    lines = [
        "# Lecture Examples",
        "",
        f"This directory contains the lecture example notebooks for {readme_title}.",
        "",
        "## Files",
        "",
    ]

    for nb_path in notebook_paths:
        key = f"{lecture_dir.name}/{nb_path.stem}"
        summary = SUMMARY_OVERRIDES.get(key, "Lecture example notebook.")
        title = extract_title(nb_path)
        py_name = f"{nb_path.stem}.py"
        lines.append(f"- `{nb_path.name}` and `{py_name}`")
        lines.append(f"  Title: {title}")
        lines.append(f"  Summary: {summary}")
        if key in OPTIONAL_NOTES:
            lines.append(f"  {OPTIONAL_NOTES[key]}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These notebooks are examples that accompany the lecture.",
            "- Each notebook has a paired `.py` script generated for local reading and version "
            "control.",
            "- Some notebooks still contain Colab install cells in the `.ipynb` source; after "
            "`uv sync`, those cells can usually be skipped locally. For lecture-specific extras "
            "such as Prophet or PyTorch, install the matching optional `uv` group first.",
            "- These examples are not the same thing as separate classroom practical sessions.",
            "- When a lecture also has a dedicated practical session, it lives in "
            "`practical_session/`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_lecture_readme(lecture_dir: Path) -> str:
    first_line = (lecture_dir / "README.md").read_text().splitlines()[0].strip()
    examples_dir = lecture_dir / "lecture_examples"
    notebook_paths = sorted(examples_dir.glob("example_*.ipynb"))

    lines = [
        first_line,
        "",
        "This directory contains the lecture materials for this topic.",
        "",
        "## Core Files",
        "",
        "- `lecture_notes.md`",
        "- `links.yaml`",
        "- `slides/lecture.pdf`",
        "- `lecture_examples/README.md`",
        "",
        "## Lecture Examples",
        "",
    ]

    for nb_path in notebook_paths:
        key = f"{lecture_dir.name}/{nb_path.stem}"
        summary = SUMMARY_OVERRIDES.get(key, "Lecture example notebook.")
        title = extract_title(nb_path)
        lines.append(
            f"- `lecture_examples/{nb_path.name}` and `lecture_examples/{nb_path.stem}.py`: "
            f"{title}. {summary}"
        )
        if key in OPTIONAL_NOTES:
            lines.append(f"  {OPTIONAL_NOTES[key]}")

    if (lecture_dir / "practical_session").exists():
        lines.extend(
            [
                "",
                "## Additional Materials",
                "",
                "- `practical_session/README.md`",
                "- `practical_session/` contains separate classroom practical materials for this "
                "lecture.",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for lecture_dir in iter_lecture_dirs():
        examples_dir = lecture_dir / "lecture_examples"
        if not examples_dir.exists():
            continue

        for nb_path in sorted(examples_dir.glob("example_*.ipynb")):
            notebook_to_percent_script(nb_path, examples_dir / f"{nb_path.stem}.py")

        (examples_dir / "README.md").write_text(build_examples_readme(lecture_dir))
        (lecture_dir / "README.md").write_text(build_lecture_readme(lecture_dir))


if __name__ == "__main__":
    main()
