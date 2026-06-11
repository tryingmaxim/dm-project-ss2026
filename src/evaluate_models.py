"""
evaluate_models.py
------------------
Post-hoc evaluation helpers for the student dropout project.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, TABLES_DIR
from src.train_models import get_models
from src.utils import ensure_dirs


def best_model_name(cv_results: pd.DataFrame, metric: str = "f1_mean") -> str:
    """Return the model name with the highest mean score for the given metric."""
    return cv_results.sort_values(metric, ascending=False).iloc[0]["model"]


def fit_and_save_best(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    models: dict | None = None,
) -> object:
    """Fit the named model on the full dataset and persist it to disk."""
    if models is None:
        models = get_models()

    pipeline = models[model_name]
    pipeline.fit(X, y)

    ensure_dirs(MODELS_DIR)
    out_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(pipeline, out_path)
    print(f"Best model saved to {out_path}")
    return pipeline


def evaluate_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    models: dict | None = None,
) -> dict:
    """Evaluate the best model via stratified cross-validation and return diagnostics."""
    if models is None:
        models = get_models()

    pipeline = models[model_name]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_y_true, all_y_pred, all_y_prob = [], [], []
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    for train_idx, test_idx in cv.split(X_arr, y_arr):
        from sklearn.base import clone

        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train, y_train)

        all_y_true.extend(y_test)
        all_y_pred.extend(fold_pipe.predict(X_test))
        all_y_prob.extend(fold_pipe.predict_proba(X_test)[:, 1])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    cm = confusion_matrix(all_y_true, all_y_pred)
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    roc_auc_val = auc(fpr, tpr)
    precision_arr, recall_arr, _ = precision_recall_curve(all_y_true, all_y_prob)
    avg_precision = average_precision_score(all_y_true, all_y_prob)
    report = classification_report(
        all_y_true,
        all_y_pred,
        target_names=["Graduate (0)", "Dropout (1)"],
    )
    report_dict = classification_report(all_y_true, all_y_pred, output_dict=True)

    print(f"\nClassification Report - {model_name}")
    print(report)
    print(f"ROC-AUC: {roc_auc_val:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    return {
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc_val,
        "precision_curve": precision_arr,
        "recall_curve": recall_arr,
        "avg_precision": avg_precision,
        "y_test": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
        "report": report,
        "report_dict": report_dict,
    }


def get_lr_coefficients(
    X: pd.DataFrame,
    y: pd.Series,
    top_features: list[str] | None = None,
) -> pd.Series:
    """Fit a standardized logistic regression model and return sorted coefficients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_use = X.copy()
    if top_features is not None:
        X_use = X_use[top_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_use)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    lr.fit(X_scaled, y)

    coefs = pd.Series(lr.coef_[0], index=X_use.columns, name="coefficient")
    return coefs.sort_values()


def descriptive_statistics(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Compute descriptive statistics for the binary dataset."""
    stats = df.describe().T
    if save:
        ensure_dirs(TABLES_DIR)
        stats.to_csv(os.path.join(TABLES_DIR, "descriptive_statistics.csv"))
    return stats
