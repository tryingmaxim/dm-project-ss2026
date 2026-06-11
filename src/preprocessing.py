"""
preprocessing.py
----------------
Builds reusable sklearn pipelines for the binary dropout classification task.

Two pipelines are returned:
  * ``build_pipeline(model)``  –  generic pipeline with StandardScaler
  * ``build_pipeline_tree(model)``  –  no scaling (for tree-based models and NB)

The distinction matters because Logistic Regression, SVM, and kNN are
sensitive to feature scale, while Decision Tree, Random Forest, and Naive
Bayes are not.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import ALL_FEATURES


# ---------------------------------------------------------------------------
# Categorical columns that carry nominal numeric codes
# (they are already integer-encoded in this dataset, so no OrdinalEncoder
#  is needed; we just pass them through as-is)
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Previous qualification",
]


def build_pipeline(model) -> Pipeline:
    """Backward-compatible alias for the default scaled pipeline."""
    return build_scaled_pipeline(model)


def build_scaled_pipeline(model, selector=None) -> Pipeline:
    """
    Build a sklearn Pipeline with median imputation, optional selection,
    and StandardScaler.

    Use this for models that are scale-sensitive:
    Logistic Regression, SVM, kNN.

    Parameters
    ----------
    model :
        A fitted or unfitted sklearn estimator.

    Returns
    -------
    Pipeline
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("clf", model))
    return Pipeline(steps)


def build_pipeline_tree(model) -> Pipeline:
    """
    Build a sklearn Pipeline with median imputation only (no scaling).

    Use this for scale-invariant models:
    Decision Tree, Random Forest, Naive Bayes.

    Parameters
    ----------
    model :
        A fitted or unfitted sklearn estimator.

    Returns
    -------
    Pipeline
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     model),
    ])


def build_one_hot_preprocessor(
    feature_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that one-hot encodes nominal integer-coded columns.

    Numeric columns are median-imputed and scaled; categorical columns are
    imputed with the most frequent value and then one-hot encoded.
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    feature_cols = [col for col in feature_cols]
    categorical_use = [col for col in categorical_cols if col in feature_cols]
    numeric_use = [col for col in feature_cols if col not in categorical_use]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_use),
            ("cat", categorical_transformer, categorical_use),
        ],
        remainder="drop",
    )


def build_one_hot_pipeline(
    model,
    feature_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> Pipeline:
    """Build a pipeline that one-hot encodes nominal coded features before fitting."""
    return Pipeline([
        ("preprocessor", build_one_hot_preprocessor(feature_cols, categorical_cols)),
        ("clf", model),
    ])


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Return columns with missing value counts (>0 only)."""
    missing = df.isnull().sum()
    return missing[missing > 0]
