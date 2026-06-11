"""
config.py
---------
Central configuration for the student dropout prediction project.
All paths, constants, and feature group definitions live here so that
every other module stays free of hard-coded values.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

RESULTS_DIR       = os.path.join(BASE_DIR, "results")
TABLES_DIR        = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR       = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR        = os.path.join(RESULTS_DIR, "models")
EXPLORATION_DIR   = os.path.join(RESULTS_DIR, "exploration")
DEFENSE_DIR       = os.path.join(RESULTS_DIR, "defense")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS = 10

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
TARGET_COL     = "Target"
TARGET_DROPOUT = "Dropout"
TARGET_GRADUATE = "Graduate"
TARGET_ENROLLED = "Enrolled"   # removed for binary classification

# Binary encoding: Dropout=1, Graduate=0
BINARY_TARGET_MAP = {TARGET_DROPOUT: 1, TARGET_GRADUATE: 0}

# ---------------------------------------------------------------------------
# Feature groups
# (column names exactly as they appear after stripping whitespace/tabs)
# ---------------------------------------------------------------------------
DEMOGRAPHIC_FEATURES = [
    "Marital status",
    "Gender",
    "Age at enrollment",
    "International",
    "Nacionality",
    "Daytime/evening attendance",
    "Application mode",
    "Application order",
]

SOCIOECONOMIC_FEATURES = [
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
]

ACADEMIC_FEATURES = [
    "Admission grade",
    "Previous qualification",
    "Previous qualification (grade)",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

MACROECONOMIC_FEATURES = [
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Other features not assigned to a named group above
OTHER_FEATURES = [
    "Course",
]

ALL_FEATURE_GROUPS = {
    "demographic":    DEMOGRAPHIC_FEATURES,
    "socioeconomic":  SOCIOECONOMIC_FEATURES,
    "academic":       ACADEMIC_FEATURES,
    "macroeconomic":  MACROECONOMIC_FEATURES,
    "other":          OTHER_FEATURES,
}

# Full ordered feature list used for modelling
ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + SOCIOECONOMIC_FEATURES
    + ACADEMIC_FEATURES
    + MACROECONOMIC_FEATURES
    + OTHER_FEATURES
)

# Top-N feature subsets used for model comparison
TOP_N_FEATURES = [10, 15]

# ---------------------------------------------------------------------------
# Model names (used as dict keys throughout the project)
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
    "SVM",
    "kNN",
]
