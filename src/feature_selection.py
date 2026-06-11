"""
feature_selection.py
--------------------
Computes feature importance rankings using three complementary methods:
  1. Mutual Information (information gain)
  2. ANOVA F-score (univariate filter)
  3. Random Forest feature importance (embedded method)

Results are returned as sorted DataFrames and optionally saved to CSV.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.config import (
    RANDOM_STATE,
    TABLES_DIR,
    ALL_FEATURE_GROUPS,
    TOP_N_FEATURES,
)
from src.utils import ensure_dirs


# ---------------------------------------------------------------------------
# Individual methods
# ---------------------------------------------------------------------------

def compute_mutual_info(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute mutual information scores between each feature and the target.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric, no NaN).
    y : pd.Series
        Binary target vector.

    Returns
    -------
    pd.DataFrame
        Columns: feature, mi_score – sorted descending.
    """
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    mi_scores = mutual_info_classif(X_imp, y, random_state=RANDOM_STATE)
    df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
    return df.sort_values("mi_score", ascending=False).reset_index(drop=True)


def compute_fscore(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute ANOVA F-scores and p-values for each feature.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric, no NaN).
    y : pd.Series
        Binary target vector.

    Returns
    -------
    pd.DataFrame
        Columns: feature, f_score, p_value – sorted by f_score descending.
    """
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    f_scores, p_values = f_classif(X_imp, y)
    df = pd.DataFrame({
        "feature": X.columns,
        "f_score": f_scores,
        "p_value": p_values,
    })
    return df.sort_values("f_score", ascending=False).reset_index(drop=True)


def compute_rf_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute feature importances from a Random Forest classifier.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    pd.DataFrame
        Columns: feature, rf_importance – sorted descending.
    """
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf.fit(X_imp, y)

    df = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": rf.feature_importances_,
    })
    return df.sort_values("rf_importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combined ranking
# ---------------------------------------------------------------------------

def compute_all_importances(
    X: pd.DataFrame,
    y: pd.Series,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run all three feature importance methods and return a dict of DataFrames.

    Optionally saves each result to results/tables/.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    save : bool
        If True, save CSVs to TABLES_DIR.

    Returns
    -------
    dict with keys 'mutual_info', 'fscore', 'rf_importance'
    """
    print("Computing Mutual Information scores ...")
    mi  = compute_mutual_info(X, y)

    print("Computing ANOVA F-scores ...")
    fsc = compute_fscore(X, y)

    print("Computing Random Forest feature importances ...")
    rfi = compute_rf_importance(X, y)

    results = {
        "mutual_info":  mi,
        "fscore":       fsc,
        "rf_importance": rfi,
    }

    if save:
        ensure_dirs(TABLES_DIR)
        mi.to_csv(os.path.join(TABLES_DIR, "feature_importance_mutual_info.csv"), index=False)
        fsc.to_csv(os.path.join(TABLES_DIR, "feature_importance_fscore.csv"), index=False)
        rfi.to_csv(os.path.join(TABLES_DIR, "feature_importance_rf.csv"), index=False)

    return results


def get_top_features(importance_df: pd.DataFrame, score_col: str, n: int) -> list[str]:
    """
    Return the top-N feature names from an importance DataFrame.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of one of the compute_* functions.
    score_col : str
        Column name containing the importance score.
    n : int
        Number of top features to return.

    Returns
    -------
    list of str
    """
    return importance_df.nlargest(n, score_col)["feature"].tolist()


def group_summary(
    importance_df: pd.DataFrame,
    score_col: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Aggregate mean importance scores per feature group.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must contain a 'feature' column and *score_col*.
    score_col : str
    save : bool

    Returns
    -------
    pd.DataFrame
        Columns: group, mean_score, n_features – sorted by mean_score descending.
    """
    rows = []
    for group, features in ALL_FEATURE_GROUPS.items():
        subset = importance_df[importance_df["feature"].isin(features)]
        if subset.empty:
            continue
        rows.append({
            "group": group,
            "mean_score": subset[score_col].mean(),
            "n_features": len(subset),
        })

    summary = pd.DataFrame(rows).sort_values("mean_score", ascending=False)

    if save:
        ensure_dirs(TABLES_DIR)
        summary.to_csv(
            os.path.join(TABLES_DIR, "grouped_feature_summary.csv"), index=False
        )

    return summary
