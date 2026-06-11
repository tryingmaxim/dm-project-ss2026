"""
data_loader.py
--------------
Loads and validates the raw dataset, and prepares the binary target variable
(Dropout vs. Graduate) by removing 'Enrolled' records.
"""

import pandas as pd

from src.config import (
    DATA_PATH,
    TARGET_COL,
    TARGET_ENROLLED,
    BINARY_TARGET_MAP,
    ALL_FEATURES,
)


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV file.

    The dataset uses semicolons as separators.  Column names are stripped of
    leading/trailing whitespace and tab characters after loading.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with cleaned column names.
    """
    df = pd.read_csv(path, sep=";")
    # Strip whitespace / tab artefacts from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a concise overview of the raw dataframe."""
    print(f"Shape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")


def prepare_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to Dropout / Graduate rows only and encode the binary target.

    Dropout  → 1
    Graduate → 0

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (output of ``load_raw``).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with an integer 'Target' column (0/1).
    """
    df = df[df[TARGET_COL] != TARGET_ENROLLED].copy()
    df[TARGET_COL] = df[TARGET_COL].map(BINARY_TARGET_MAP)
    df = df.reset_index(drop=True)
    return df


def get_X_y(df: pd.DataFrame, feature_cols: list | None = None):
    """
    Split a prepared dataframe into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``prepare_binary``.
    feature_cols : list, optional
        Column names to use as features. Defaults to ALL_FEATURES from config,
        filtered to columns present in *df*.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if feature_cols is None:
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y
