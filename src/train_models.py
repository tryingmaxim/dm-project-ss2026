"""
train_models.py
---------------
Defines all classification models and runs 10-fold stratified cross-validation
for each model on the full feature set (and optionally on a top-N subset).

Results are returned as a tidy DataFrame and saved to results/tables/.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_FOLDS, DEFENSE_DIR, RANDOM_STATE, TABLES_DIR
from src.feature_selection import compute_mutual_info, get_top_features
from src.preprocessing import (
    CATEGORICAL_COLS,
    build_one_hot_pipeline,
    build_pipeline,
    build_pipeline_tree,
    build_scaled_pipeline,
)
from src.utils import ensure_dirs


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Return a dict mapping model name → fitted-ready pipeline.

    Scale-sensitive models (LR, SVM, kNN) get a StandardScaler step.
    Scale-invariant models (DT, RF, NB) use a plain imputer pipeline.
    """
    models = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
        ),
        "Decision Tree": build_pipeline_tree(
            DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=None)
        ),
        "Random Forest": build_pipeline_tree(
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "Naive Bayes": build_pipeline_tree(
            GaussianNB()
        ),
        "SVM": build_pipeline(
            SVC(kernel="rbf", cache_size=500, random_state=RANDOM_STATE)
        ),
        "kNN": build_pipeline(
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        ),
    }
    return models


def build_lr_selector_pipeline(k: int = 10):
    """Build leakage-free logistic regression with per-fold MI feature selection."""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    return build_scaled_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0),
        selector=selector,
    )


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

SCORING = {
    "accuracy":  "accuracy",
    "precision": "precision",
    "recall":    "recall",
    "f1":        "f1",
    "roc_auc":   "roc_auc",
}


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict | None = None,
    label: str = "all_features",
    save: bool = True,
) -> pd.DataFrame:
    """
    Run stratified 10-fold cross-validation for every model in *models*.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (may contain NaN; pipelines handle imputation).
    y : pd.Series
        Binary target vector.
    models : dict, optional
        Name → pipeline dict.  Defaults to ``get_models()``.
    label : str
        Short descriptor used in the output filename (e.g. 'all_features',
        'top10').
    save : bool
        If True, save results to results/tables/.

    Returns
    -------
    pd.DataFrame
        One row per model; mean and std for each metric.
    """
    if models is None:
        models = get_models()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fold_scores = {}
    for name, pipeline in models.items():
        print(f"  Cross-validating: {name} ...")
        cv_res = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=1,
        )
        row = {"model": name}
        for metric in SCORING:
            vals = cv_res[f"test_{metric}"]
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"]  = np.std(vals)
        fold_scores[name] = {
            metric: cv_res[f"test_{metric}"].tolist()
            for metric in SCORING
        }
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)

    if save:
        ensure_dirs(TABLES_DIR)
        fname = f"model_comparison_{label}.csv"
        results.to_csv(os.path.join(TABLES_DIR, fname), index=False)
        fold_scores_name = f"cv_fold_scores_{label}.json"
        with open(os.path.join(TABLES_DIR, fold_scores_name), "w", encoding="utf-8") as f:
            json.dump(fold_scores, f, indent=2)
        # Always keep a canonical "model_comparison.csv" for the main run
        if label == "all_features":
            results.to_csv(os.path.join(TABLES_DIR, "model_comparison.csv"), index=False)
            with open(os.path.join(TABLES_DIR, "cv_fold_scores.json"), "w", encoding="utf-8") as f:
                json.dump(fold_scores, f, indent=2)

    return results


def compare_global_vs_pipeline_selection(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compare global Top-k MI feature selection against leakage-free per-fold selection.
    """
    mi_scores = compute_mutual_info(X, y)
    top_features = get_top_features(mi_scores, "mi_score", k)

    models = {
        "Global MI Top-10 before CV": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
        ),
        "Pipeline SelectKBest per fold": build_lr_selector_pipeline(k=k),
    }

    results_global = run_cv(
        X[top_features],
        y,
        models={"Global MI Top-10 before CV": models["Global MI Top-10 before CV"]},
        label="global_mi_top10_lr",
        save=False,
    )
    results_pipeline = run_cv(
        X,
        y,
        models={"Pipeline SelectKBest per fold": models["Pipeline SelectKBest per fold"]},
        label="pipeline_selectkbest_lr",
        save=False,
    )

    comparison = pd.concat([results_global, results_pipeline], ignore_index=True)
    f1_global = comparison.loc[
        comparison["model"] == "Global MI Top-10 before CV", "f1_mean"
    ].iloc[0]
    f1_pipeline = comparison.loc[
        comparison["model"] == "Pipeline SelectKBest per fold", "f1_mean"
    ].iloc[0]
    comparison["f1_difference_vs_pipeline_pp"] = (
        (comparison["f1_mean"] - f1_pipeline) * 100
    ).round(3)
    comparison["top_features"] = ""
    comparison.loc[
        comparison["model"] == "Global MI Top-10 before CV", "top_features"
    ] = ", ".join(top_features)
    comparison.loc[
        comparison["model"] == "Pipeline SelectKBest per fold", "top_features"
    ] = "Selected independently inside each training fold"
    comparison["interpretation"] = [
        (
            f"Potential optimistic bias vs leakage-free pipeline: "
            f"{(f1_global - f1_pipeline) * 100:.2f} percentage points."
        ),
        "Leakage-free reference estimate.",
    ]

    if save:
        ensure_dirs(DEFENSE_DIR)
        comparison.to_csv(
            os.path.join(DEFENSE_DIR, "global_top10_vs_pipeline_selector.csv"),
            index=False,
        )

    return comparison


def run_svm_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Tune the RBF SVM while explicitly keeping sklearn's gamma='scale' in the grid."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(SVC(kernel="rbf", cache_size=500, random_state=RANDOM_STATE))
    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01, 0.1, 1],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"},
        refit="f1",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X, y)

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_f1"
    )[
        [
            "param_clf__C",
            "param_clf__gamma",
            "mean_test_f1",
            "std_test_f1",
            "mean_test_roc_auc",
            "mean_test_accuracy",
            "rank_test_f1",
        ]
    ].reset_index(drop=True)
    best_summary = {
        "best_C": search.best_params_["clf__C"],
        "best_gamma": search.best_params_["clf__gamma"],
        "best_f1": float(search.best_score_),
        "best_roc_auc": float(
            cv_results.loc[cv_results["rank_test_f1"] == 1, "mean_test_roc_auc"].iloc[0]
        ),
        "best_accuracy": float(
            cv_results.loc[cv_results["rank_test_f1"] == 1, "mean_test_accuracy"].iloc[0]
        ),
    }

    if save:
        ensure_dirs(DEFENSE_DIR)
        cv_results.to_csv(
            os.path.join(DEFENSE_DIR, "svm_grid_search_results.csv"),
            index=False,
        )
        pd.DataFrame([best_summary]).to_csv(
            os.path.join(DEFENSE_DIR, "svm_grid_search_best_params.csv"),
            index=False,
        )

    return cv_results, best_summary


def compare_integer_vs_one_hot(
    X: pd.DataFrame,
    y: pd.Series,
    save: bool = True,
) -> pd.DataFrame:
    """Compare integer-coded nominal features against one-hot encoded pipelines."""
    models = {
        "Logistic Regression - Integer Codes": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
        ),
        "Logistic Regression - One-Hot": build_one_hot_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0),
            feature_cols=X.columns.tolist(),
            categorical_cols=CATEGORICAL_COLS,
        ),
        "SVM - Integer Codes": build_pipeline(
            SVC(kernel="rbf", cache_size=500, random_state=RANDOM_STATE)
        ),
        "SVM - One-Hot": build_one_hot_pipeline(
            SVC(kernel="rbf", cache_size=500, random_state=RANDOM_STATE),
            feature_cols=X.columns.tolist(),
            categorical_cols=CATEGORICAL_COLS,
        ),
    }
    comparison = run_cv(X, y, models=models, label="onehot_vs_integer", save=False)
    baseline_scores = {
        "Logistic Regression": comparison.loc[
            comparison["model"] == "Logistic Regression - Integer Codes", "f1_mean"
        ].iloc[0],
        "SVM": comparison.loc[
            comparison["model"] == "SVM - Integer Codes", "f1_mean"
        ].iloc[0],
    }
    comparison["delta_vs_integer_pp"] = 0.0
    for base_name, baseline in baseline_scores.items():
        mask = comparison["model"].str.startswith(base_name)
        comparison.loc[mask, "delta_vs_integer_pp"] = (
            (comparison.loc[mask, "f1_mean"] - baseline) * 100
        ).round(3)

    if save:
        ensure_dirs(DEFENSE_DIR)
        comparison.to_csv(
            os.path.join(DEFENSE_DIR, "onehot_vs_integer_comparison.csv"),
            index=False,
        )

    return comparison
