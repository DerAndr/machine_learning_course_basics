from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LECTURES_ROOT = REPO_ROOT / "lectures"

BASE_REQUIRED_MODULES = {
    "alibi": "alibi",
    "altair": "altair",
    "bokeh": "bokeh",
    "category_encoders": "category_encoders",
    "dtale": "dtale",
    "fairlearn": "fairlearn",
    "imblearn": "imblearn",
    "interpret": "interpret",
    "joblib": "joblib",
    "lime": "lime",
    "matplotlib": "matplotlib",
    "mlxtend": "mlxtend",
    "mord": "mord",
    "nbformat": "nbformat",
    "numpy": "numpy",
    "openml": "openml",
    "pandas": "pandas",
    "plotly": "plotly",
    "plotnine": "plotnine",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "shap": "shap",
    "skfuzzy": "skfuzzy",
    "sklearn": "sklearn",
    "statsmodels": "statsmodels",
    "sweetviz": "sweetviz",
    "ucimlrepo": "ucimlrepo",
    "umap": "umap",
    "ydata_profiling": "ydata_profiling",
}

OPTIONAL_GROUPS = {
    "ensembles": {
        "modules": {
            "catboost": "catboost",
            "lightgbm": "lightgbm",
            "xgboost": "xgboost",
        },
        "install_hint": "uv sync --group ensembles",
    },
    "time_series": {
        "modules": {
            "prophet": "prophet",
            "xgboost": "xgboost",
        },
        "install_hint": "uv sync --group time_series",
    },
    "neural_networks": {
        "modules": {
            "torch": "torch",
            "torchvision": "torchvision",
        },
        "install_hint": "uv sync --group neural_networks",
    },
    "hpo_automl": {
        "modules": {
            "h2o": "h2o",
            "hyperopt": "hyperopt",
            "optuna": "optuna",
            "skopt": "scikit-optimize",
        },
        "install_hint": "uv sync --group hpo_automl",
    },
    "xai_piml": {
        "modules": {
            "piml": "piml",
        },
        "install_hint": "uv sync --group xai_piml",
    },
}


def check_module_imports(modules: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for module_name, package_name in sorted(modules.items()):
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def check_example_pairs() -> list[str]:
    missing: list[str] = []
    for nb_path in sorted(LECTURES_ROOT.glob("lecture_*/lecture_examples/example_*.ipynb")):
        py_path = nb_path.with_suffix(".py")
        if not py_path.exists():
            missing.append(str(py_path.relative_to(REPO_ROOT)))
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the local notebook environment for this course repository."
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        choices=sorted(OPTIONAL_GROUPS),
        help="Require one optional lecture group in addition to the baseline environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_groups = args.group

    required_modules = dict(BASE_REQUIRED_MODULES)
    for group_name in requested_groups:
        required_modules.update(OPTIONAL_GROUPS[group_name]["modules"])

    missing_packages = check_module_imports(required_modules)
    missing_pairs = check_example_pairs()

    if missing_packages:
        print("Missing Python packages:")
        for package in missing_packages:
            print(f"- {package}")

    if missing_pairs:
        print("Missing paired .py files:")
        for path in missing_pairs:
            print(f"- {path}")

    optional_missing_by_group: dict[str, list[str]] = {}
    for group_name, config in OPTIONAL_GROUPS.items():
        if group_name in requested_groups:
            continue
        missing = check_module_imports(config["modules"])
        if missing:
            optional_missing_by_group[group_name] = missing

    if optional_missing_by_group:
        print("Optional lecture groups not installed:")
        for group_name, packages in optional_missing_by_group.items():
            packages_str = ", ".join(packages)
            hint = OPTIONAL_GROUPS[group_name]["install_hint"]
            print(f"- {group_name}: {packages_str}")
            print(f"  Install with: {hint}")

    if missing_packages or missing_pairs:
        return 1

    if requested_groups:
        groups_str = ", ".join(requested_groups)
        print(f"Notebook environment check passed for baseline + groups: {groups_str}.")
    else:
        print("Notebook environment check passed for the baseline environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
