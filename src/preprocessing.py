"""
MODULE 1 (part b) - Data Preprocessing
=====================================
Thesis reference: Section 3.3.2 / 4.3 - steps 5 to 8 of the preprocessing
pipeline.

Steps implemented here
-----------------------
1. Detect the data type of every column (numerical / categorical / datetime).
2. Handle missing values
       * categorical -> mode, falling back to the placeholder "Unknown"
       * numerical   -> mean or median
3. Encode categorical features (label encoding by default, one-hot optional).
4. Split the frame into features ``X`` and target ``y``.
5. Scale the numerical features with :class:`StandardScaler`.
6. Expose a single :func:`prepare_dataset` helper that chains 1-5 and returns a
   :class:`PreparedDataset` bundle used by the training module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config
from .utils import is_categorical, standardize_column_name, standardize_column_names

__all__ = [
    "PreprocessingError",
    "PreparedDataset",
    "resolve_column",
    "detect_id_columns",
    "detect_column_types",
    "handle_missing_values",
    "encode_categorical_features",
    "scale_features",
    "split_features_target",
    "prepare_dataset",
]

MissingStrategy = Literal["mean", "median"]
EncodingMethod = Literal["label", "onehot"]

# Headers that identify a row rather than describe it (``order_id``, ``posting_id``,
# ``customer id`` ...).  Such columns must never be used as model features.
ID_COLUMN_PATTERN = re.compile(r"(^|_)id($|_)|(^|_)(order|posting|user|customer|transaction|invoice|product)_?id($|_)")


class PreprocessingError(ValueError):
    """Raised for unrecoverable preprocessing problems (e.g. unknown target)."""


# --------------------------------------------------------------------------- #
# 0. Column resolution
# --------------------------------------------------------------------------- #
def resolve_column(df: pd.DataFrame, name: str) -> str:
    """
    Find the real column of ``df`` that matches ``name``.

    Accepts an exact match first, then a ``snake_case`` match, so ``"Total
    Sales"``, ``"total_sales"`` and ``"TOTAL SALES"`` all resolve to the same
    column.
    """
    if name in df.columns:
        return name

    wanted = standardize_column_name(name)
    for column in df.columns:
        if standardize_column_name(column) == wanted:
            return column
    raise PreprocessingError(
        f"Column '{name}' was not found. Available columns: {', '.join(map(str, df.columns))}"
    )


# --------------------------------------------------------------------------- #
# 0b. Identifier detection
# --------------------------------------------------------------------------- #
def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the columns that identify a row instead of describing it.

    A column is treated as an identifier when its (standardised) name looks like
    ``order_id`` / ``posting_id`` / ``customer_id``, or when it is a free-text
    column whose values are (almost) unique - such a column would let a tree
    memorise the training rows and destroy generalisation.
    """
    identifiers: list[str] = []
    rows = len(df)

    for column in df.columns:
        name = standardize_column_name(column)
        if ID_COLUMN_PATTERN.search(name):
            identifiers.append(column)
            continue
        if rows > 20 and not pd.api.types.is_numeric_dtype(df[column]):
            ratio = df[column].nunique(dropna=False) / rows
            if ratio >= config.UNIQUE_VALUE_RATIO:
                identifiers.append(column)

    return identifiers


# --------------------------------------------------------------------------- #
# 1. Type detection
# --------------------------------------------------------------------------- #
def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Classify every column as ``numeric``, ``categorical`` or ``datetime``.

    Mirrors step 5 of the report - "the system automatically infers the data
    types of each column (numerical, categorical, text, etc.)".
    """
    detected: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": []}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            detected["datetime"].append(column)
        elif is_categorical(series):
            detected["categorical"].append(column)
        else:
            detected["numeric"].append(column)
    return detected


# --------------------------------------------------------------------------- #
# 2. Missing values
# --------------------------------------------------------------------------- #
def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: MissingStrategy = "mean",
    categorical_placeholder: str = config.CATEGORICAL_PLACEHOLDER,
    drop_empty_rows: bool = False,
) -> pd.DataFrame:
    """
    Impute missing values column by column.

    * Categorical columns receive their mode (most frequent value); entirely
      empty columns receive ``categorical_placeholder``.
    * Numerical columns receive their mean ("mean") or median ("median").
    * Datetime columns receive their most frequent timestamp.
    """
    if numeric_strategy not in ("mean", "median"):
        raise PreprocessingError("numeric_strategy must be 'mean' or 'median'.")

    frame = df.copy()
    if drop_empty_rows:
        frame = frame.dropna(how="all")

    types = detect_column_types(frame)

    for column in types["categorical"]:
        series = frame[column]
        mode = series.mode(dropna=True)
        replacement: Any = mode.iloc[0] if not mode.empty else categorical_placeholder
        if pd.isna(replacement):
            replacement = categorical_placeholder
        frame[column] = series.fillna(replacement)

    for column in types["numeric"]:
        series = frame[column]
        replacement = series.mean() if numeric_strategy == "mean" else series.median()
        if pd.isna(replacement):
            replacement = 0.0
        frame[column] = series.fillna(replacement)

    for column in types["datetime"]:
        series = frame[column]
        mode = series.mode(dropna=True)
        if not mode.empty:
            frame[column] = series.fillna(mode.iloc[0])

    return frame


# --------------------------------------------------------------------------- #
# 3. Encoding
# --------------------------------------------------------------------------- #
def encode_categorical_features(
    df: pd.DataFrame,
    method: EncodingMethod = "label",
    exclude: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Encode categorical (and datetime) columns into numeric form.

    Returns the transformed frame **and** the fitted encoder per column, so the
    exact same transformation can be replayed at prediction time.
    """
    if method not in ("label", "onehot"):
        raise PreprocessingError("method must be 'label' or 'onehot'.")

    frame = df.copy()
    encoders: dict[str, Any] = {}
    excluded = set(exclude)

    targets: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            targets.append(column)
        elif is_categorical(frame[column]):
            targets.append(column)

    for column in targets:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            values = pd.to_datetime(frame[column]).astype("int64").astype(str)
        else:
            values = frame[column].astype(str)

        if method == "label":
            encoder = LabelEncoder()
            frame[column] = encoder.fit_transform(values)
            encoders[column] = {"type": "label", "encoder": encoder}
        else:
            categories = sorted(values.unique().tolist())
            mapping = {category: index for index, category in enumerate(categories)}
            frame[column] = values.map(mapping).astype(float)
            encoders[column] = {"type": "onehot", "categories": categories, "mapping": mapping}

    return frame, encoders


# --------------------------------------------------------------------------- #
# 4. Feature / target split
# --------------------------------------------------------------------------- #
def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for ``target_column``, dropping ``drop_columns`` from ``X``."""
    target = resolve_column(df, target_column)
    features = df.drop(columns=[target])
    unwanted = [c for c in drop_columns if c in features.columns]
    if unwanted:
        features = features.drop(columns=unwanted)
    if features.shape[1] == 0:
        raise PreprocessingError("No feature columns remain after dropping the target/ID columns.")
    return features, df[target]


# --------------------------------------------------------------------------- #
# 5. Feature scaling
# --------------------------------------------------------------------------- #
def scale_features(
    x_train: pd.DataFrame | np.ndarray,
    x_test: pd.DataFrame | np.ndarray | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """
    Standardise ``x_train`` (and optionally ``x_test``) with a ``StandardScaler``.

    The scaler is fitted on the training split only - never on the test split -
    which prevents data leakage.
    """
    fitted = scaler if scaler is not None else StandardScaler()
    x_train_scaled = fitted.fit_transform(np.asarray(x_train, dtype=float))

    if x_test is None:
        return x_train_scaled, None, fitted

    x_test_scaled = fitted.transform(np.asarray(x_test, dtype=float))
    return x_train_scaled, x_test_scaled, fitted


# --------------------------------------------------------------------------- #
# 6. Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PreparedDataset:
    """Everything the training module needs after preprocessing."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    x_full: np.ndarray                 # encoded **and scaled** feature matrix
    y_full: pd.Series
    feature_names: list[str]
    target_name: str
    scaler: StandardScaler
    encoders: dict[str, Any] = field(default_factory=dict)
    label_encoder: LabelEncoder | None = None

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_classes(self) -> int:
        return int(pd.Series(self.y_full).nunique())

    @property
    def class_labels(self) -> list[str]:
        if self.label_encoder is not None:
            return [str(c) for c in self.label_encoder.classes_]
        return [str(c) for c in sorted(pd.Series(self.y_full).unique().tolist())]

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target_name,
            "features": self.n_features,
            "classes": self.n_classes,
            "train_rows": int(np.asarray(self.x_train).shape[0]),
            "test_rows": int(np.asarray(self.x_test).shape[0]),
        }


def prepare_dataset(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Sequence[str] = config.DEFAULT_DROP_COLUMNS,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    numeric_strategy: MissingStrategy = "mean",
    encoding_method: EncodingMethod = "label",
    auto_drop_ids: bool = True,
) -> PreparedDataset:
    """
    Full preprocessing pipeline (steps 1-5 of the module).

    The target is label-encoded automatically when it is categorical, which is
    the usual case for a survey dataset (e.g. ``status = Completed/Pending/
    Cancelled``).  Missing values are imputed, categorical columns are encoded,
    and the features are scaled with :class:`StandardScaler`.

    Column headers are standardised first (``"Order Id"`` -> ``"order_id"``) so
    that identifiers listed in ``drop_columns`` are always removed.  When
    ``auto_drop_ids`` is true, any further identifier-like column detected by
    :func:`detect_id_columns` is dropped as well.
    """
    frame = standardize_column_names(df)
    target = resolve_column(frame, target_column)

    dropped: list[str] = [column for column in (drop_columns or ()) if column in frame.columns]
    if auto_drop_ids:
        dropped.extend(column for column in detect_id_columns(frame) if column != target)

    cleaned = handle_missing_values(frame, numeric_strategy=numeric_strategy)
    x_raw, y_raw = split_features_target(cleaned, target, dropped)

    x_encoded, encoders = encode_categorical_features(x_raw, method=encoding_method)

    label_encoder: LabelEncoder | None = None
    if is_categorical(y_raw) or not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y_encoded: pd.Series = pd.Series(
            label_encoder.fit_transform(y_raw.astype(str)),
            name=str(target),
            index=y_raw.index,
        )
    else:
        y_encoded = y_raw.astype(float)

    x = x_encoded.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=test_size, random_state=random_state
    )
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

    # The winner is refit on the *whole* dataset before it is persisted, so the
    # full matrix must live in the same space as the train/test splits (scaled).
    # Otherwise a reloaded bundle would mis-predict, because `encode_for_bundle`
    # replays `scaler.transform` on every incoming row.
    x_full = scaler.transform(x.values.astype(float))

    return PreparedDataset(
        x_train=x_train_scaled,
        x_test=x_test_scaled,
        y_train=y_train,
        y_test=y_test,
        x_full=x_full,
        y_full=y_encoded,
        feature_names=list(x.columns),
        target_name=str(target),
        scaler=scaler,
        encoders=encoders,
        label_encoder=label_encoder,
    )
