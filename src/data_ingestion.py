"""
MODULE 1 (part a) - Data Ingestion
=================================
Thesis reference: Section 3.3.1 / 4.3 / 4.3.1 - "Data Ingestion and Data
Preprocessing module".

Responsibilities
------------------
1. Accept a CSV / Excel file from the Streamlit uploader or a filesystem path.
2. Validate the file format and report a friendly error for anything else.
3. Read the file into a :class:`pandas.DataFrame`.
4. Standardise the column headers ("Total Sales ($)" -> "total_sales").
5. Replace blank / duplicate headers with default names.
6. Persist the cleaned frame as ``source_data.csv`` for the later stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from . import config
from .utils import ensure_directory, get_file_extension, is_categorical, standardize_column_names

__all__ = [
    "DataIngestionError",
    "is_supported_file",
    "load_dataset",
    "ingest_dataset",
    "save_source_data",
    "preview_dataset",
]


class DataIngestionError(ValueError):
    """Raised when a file cannot be ingested (unsupported type or unreadable)."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def is_supported_file(filename: str | Path) -> bool:
    """True when the extension of ``filename`` is CSV / XLSX / XLS."""
    return get_file_extension(filename) in config.SUPPORTED_EXTENSIONS


def _read_dataframe(source: Any, extension: str) -> pd.DataFrame:
    """Read ``source`` (path or file-like object) into a DataFrame."""
    try:
        if extension == "csv":
            return pd.read_csv(source)
        if extension in ("xlsx", "xls"):
            return pd.read_excel(source, engine="openpyxl")
    except Exception as exc:  # pragma: no cover - depends on corrupt input
        raise DataIngestionError(f"Could not read the file: {exc}") from exc

    raise DataIngestionError(
        f"Unsupported file format '.{extension}'. "
        f"Supported formats: {', '.join(config.SUPPORTED_EXTENSIONS)}."
    )


def load_dataset(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """
    Load a dataset from a path or an uploaded file-like object.

    Parameters
    ----------
    source:
        Either a filesystem path or an object exposing ``.read()`` (the object
        returned by ``streamlit.file_uploader``).
    filename:
        Original file name, required when ``source`` is a file-like object that
        does not carry a ``.name`` attribute.
    """
    if hasattr(source, "name") and filename is None:
        filename = getattr(source, "name", None)

    if filename is None and isinstance(source, (str, Path)):
        filename = str(source)

    if not filename:
        raise DataIngestionError("The uploaded file name could not be determined.")

    extension = get_file_extension(filename)
    if extension not in config.SUPPORTED_EXTENSIONS:
        raise DataIngestionError(
            f"Unsupported file format '.{extension}'. "
            f"Supported formats: {', '.join(config.SUPPORTED_EXTENSIONS)}."
        )
    return _read_dataframe(source, extension)


# --------------------------------------------------------------------------- #
# Cleaning / persistence
# --------------------------------------------------------------------------- #
def save_source_data(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist ``df`` as ``source_data.csv`` and return the written path."""
    target = Path(path) if path is not None else config.SOURCE_DATA_PATH
    ensure_directory(target.parent)
    df.to_csv(target, index=False)
    return target


def preview_dataset(df: pd.DataFrame, rows: int = 5) -> dict[str, Any]:
    """Small summary dictionary used by the UI and by the unit tests."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head(rows).to_dict(orient="records"),
        "memory_usage": df.memory_usage(deep=True).sum(),
    }


def ingest_dataset(
    source: str | Path | BinaryIO,
    filename: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Full ingestion pipeline: validate -> read -> standardise headers -> save.

    Returns
    -------
    (df, saved_path)
        ``df``           - cleaned DataFrame (headers standardised)
        ``saved_path``   - location of the persisted ``source_data.csv``
    """
    raw = load_dataset(source, filename=filename)
    if raw.empty:
        raise DataIngestionError("The uploaded dataset is empty.")

    cleaned = standardize_column_names(raw)
    # Categorical gaps are filled with a placeholder so that no information is
    # lost at ingestion time (thesis S4.3 - "filling missing values ... Unknown").
    # Numeric gaps are left untouched so that `preprocessing.handle_missing_values`
    # can impute them with the mean/median of a genuine numeric column.
    for column in cleaned.columns:
        if is_categorical(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(config.CATEGORICAL_PLACEHOLDER)

    saved_path = save_source_data(cleaned, save_path)
    return cleaned, saved_path
