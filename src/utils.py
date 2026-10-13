"""
Small, dependency-free helpers shared by every module.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column-name standardisation  (thesis S4.3 step 4)
# --------------------------------------------------------------------------- #
_NON_WORD = re.compile(r"[^\w\s]+")
_MULTI_UNDERSCORE = re.compile(r"_{2,}")


def standardize_column_name(name: Any) -> str:
    """
    Convert a raw column header into a clean ``snake_case`` identifier.

    ``" Total Sales ($) "`` -> ``"total_sales"``

    Rules (matching the report): strip surrounding white space, lower-case,
    drop special characters, replace spaces with a single underscore and
    collapse repeated underscores.
    """
    text = "" if name is None else str(name)
    text = text.strip().lower()
    text = _NON_WORD.sub(" ", text)          # remove punctuation / symbols
    text = text.replace(" ", "_")           # spaces -> underscore
    text = _MULTI_UNDERSCORE.sub("_", text)  # collapse duplicates
    return text.strip("_")


def standardize_column_names(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Apply :func:`standardize_column_name` to every column of ``df``.

    Empty / duplicated headers are replaced with ``column_<i>`` so that the
    downstream pipeline always receives unique, valid names.
    """
    frame = df if inplace else df.copy()
    seen: dict[str, int] = {}
    new_columns: list[str] = []

    for position, original in enumerate(frame.columns):
        candidate = standardize_column_name(original)
        if not candidate:
            candidate = f"column_{position}"
        if candidate in seen:
            seen[candidate] += 1
            candidate = f"{candidate}_{seen[candidate]}"
        else:
            seen[candidate] = 0
        new_columns.append(candidate)

    frame.columns = new_columns
    return frame


# --------------------------------------------------------------------------- #
# File / path helpers
# --------------------------------------------------------------------------- #
def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` as a directory (recursively) and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_extension(filename: str | Path) -> str:
    """Return the lower-case extension of ``filename`` without the leading dot."""
    return Path(str(filename)).suffix.lower().lstrip(".")


def safe_filename(name: str) -> str:
    """Filesystem-safe stem: ``"Random Forest"`` -> ``"Random_Forest"``."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


# --------------------------------------------------------------------------- #
# DataFrame helpers
# --------------------------------------------------------------------------- #
def is_categorical(series: pd.Series) -> bool:
    """True when ``series`` should be treated as categorical."""
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return False


def is_continuous_numeric(series: pd.Series) -> bool:
    """
    True when ``series`` holds *continuous* numbers rather than class labels.

    This mirrors the rule scikit-learn applies to a target (``type_of_target``):
    a numeric column counts as continuous as soon as one of its values is not a
    whole number.  Every estimator of the model zoo is a classifier, and a
    classifier refuses such a target with

        ValueError: Unknown label type: 'continuous'

    so the dashboard and ``model_training.run_training_pipeline`` use this helper
    to name the offending column *before* training, instead of letting thirteen
    candidates fail and reporting only "every candidate model failed".

    Integer-coded labels (``0`` / ``1`` / ``2``), booleans and non-numeric columns
    are never reported as continuous.
    """
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return False

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:  # an empty / all-missing column is not "continuous"
        return False
    return bool(np.any(finite != np.floor(finite)))


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Names of the numeric columns of ``df`` (excluding ``exclude``)."""
    excluded = set(exclude)
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Names of the categorical columns of ``df`` (excluding ``exclude``)."""
    excluded = set(exclude)
    return [c for c in df.columns if c not in excluded and is_categorical(df[c])]


def dataframe_to_html(df: pd.DataFrame, title: str = "Data Preview", max_rows: int = 20) -> str:
    """Render a small DataFrame as a self-contained HTML snippet (fallback EDA report)."""
    return (
        f"<h3>{title}</h3>"
        f"<p>Shape: {df.shape[0]} rows x {df.shape[1]} columns</p>"
        + df.head(max_rows).to_html(border=1, classes="table table-striped", escape=True)
    )


def human_readable_bytes(size: float) -> str:
    """Format a byte count using B / KB / MB / GB units."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def dataframe_memory_usage(df: pd.DataFrame) -> str:
    """Human readable in-memory footprint of ``df`` (matches the EDA report wording)."""
    return human_readable_bytes(float(df.memory_usage(deep=True).sum()))


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def round_metrics(values: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    """Round every numeric value of ``values``; leave non-numeric values untouched."""
    rounded: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
            rounded[key] = round(float(value), digits)
        else:
            rounded[key] = value
    return rounded


def format_metrics(metrics: dict[str, Any], digits: int = 4) -> str:
    """Pretty one-line summary of a metrics dictionary."""
    return " | ".join(
        f"{k}: {v:.{digits}f}" if isinstance(v, (int, float, np.floating)) else f"{k}: {v}"
        for k, v in metrics.items()
    )


def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a text file returning ``""`` when the file does not exist."""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    with io.open(file_path, "r", encoding=encoding) as handle:
        return handle.read()
