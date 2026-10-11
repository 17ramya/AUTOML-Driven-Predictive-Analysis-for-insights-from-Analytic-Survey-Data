"""
Project-wide configuration: paths, hyper-parameters and constants.

Keeping every path / constant in one module means the Streamlit app, the CLI
(``main.py``) and the unit tests all agree on where artefacts are written.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility / modelling constants
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42          # used everywhere a random seed is accepted
TEST_SIZE: float = 0.2          # 80 % train - 20 % test split (thesis S4.4.1)
CV_FOLDS: int = 3               # cv used by GridSearchCV in the thesis code
CV_FOLDS_CROSS_VAL: int = 5     # cv used by cross_val_score in the thesis code

# Target column of the sample survey dataset (Table 4.2 of the report).
DEFAULT_TARGET: str = "status"
CATEGORICAL_PLACEHOLDER: str = "Unknown"
SUPPORTED_EXTENSIONS: tuple[str, ...] = ("csv", "xlsx", "xls")

# Columns present in the sample dataset that must never be used as features.
DEFAULT_DROP_COLUMNS: tuple[str, ...] = ("order_id", "order_date")

# A free-text column is treated as a row identifier when at least this fraction
# of its values are distinct (see `preprocessing.detect_id_columns`).
UNIQUE_VALUE_RATIO: float = 0.95

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
DOCS_DIR: Path = BASE_DIR / "docs"

# Artefacts produced at runtime
SOURCE_DATA_PATH: Path = DATA_DIR / "source_data.csv"
EDA_REPORT_PATH: Path = REPORTS_DIR / "eda_report.html"
BUNDLE_PATH: Path = MODELS_DIR / "automl_bundle.pkl"
COMPARISON_CSV_PATH: Path = REPORTS_DIR / "model_comparison.csv"

# Bundled sample datasets (created by ``scripts/make_sample_data.py``)
SAMPLE_TABULAR_PATH: Path = SAMPLE_DATA_DIR / "amazon_sales_data_2025.csv"
SAMPLE_LISTINGS_PATH: Path = SAMPLE_DATA_DIR / "product_listings.csv"
SAMPLE_IMAGES_DIR: Path = SAMPLE_DATA_DIR / "images"


def ensure_directories() -> None:
    """Create every runtime directory if it does not exist yet."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        DOCS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Plot / UI constants
# --------------------------------------------------------------------------- #
METRIC_ORDER: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "mcc",
    "kappa",
    "auc",
    "training_time",
    "score",
)

# Human readable labels used in charts and tables.
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "mcc": "MCC",
    "kappa": "Kappa",
    "auc": "AUC",
    "training_time": "Training Time (s)",
    "score": "Overall Score",
}

# Weights used to build the composite "Overall Score" reported in Figure 5.12.
SCORE_WEIGHTS: dict[str, float] = {
    "accuracy": 0.25,
    "f1_score": 0.25,
    "precision": 0.15,
    "recall": 0.15,
    "auc": 0.10,
    "kappa": 0.05,
    "mcc": 0.05,
}
