"""
PREDICTION / INFERENCE
=====================
Thesis reference: Sections 3.3.3 (final step) and 5.3 - "The best model is saved
and can be downloaded for future use. Users can also load and compare this model
with others to see which performs better."

The helpers below replay the *exact* preprocessing that was applied during
training (encoders + scaler from the saved bundle) so that a raw CSV can be
scored without any manual preparation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import config
from .model_training import load_bundle, load_model
from .preprocessing import resolve_column
from .utils import round_metrics, standardize_column_names

__all__ = [
    "PredictionError",
    "encode_for_bundle",
    "predict_with_bundle",
    "predict_dataframe",
    "model_metadata",
    "evaluate_predictions",
    "compare_saved_models",
]


class PredictionError(RuntimeError):
    """Raised when a saved model cannot be applied to the supplied data."""


# --------------------------------------------------------------------------- #
# Preprocessing replay
# --------------------------------------------------------------------------- #
def _apply_encoders(df: pd.DataFrame, encoders: dict[str, Any]) -> pd.DataFrame:
    """Apply the stored label/one-hot encoders to the columns they were fitted on."""
    frame = df.copy()
    for column, spec in encoders.items():
        if column not in frame.columns:
            continue

        if spec.get("type") == "label":
            encoder = spec["encoder"]
            known = {str(c): index for index, c in enumerate(encoder.classes_)}
            frame[column] = frame[column].astype(str).map(known).fillna(-1).astype(float)
        else:
            mapping = spec.get("mapping", {})
            frame[column] = frame[column].astype(str).map(mapping).fillna(-1.0).astype(float)

    return frame


def encode_for_bundle(df: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    """
    Convert a raw DataFrame into the scaled feature matrix expected by the model.

    Steps: standardise headers -> drop target/ID columns -> replay encoders ->
    reindex to the training column order -> coerce to numeric -> scale.
    """
    feature_names: list[str] = list(bundle.get("feature_names") or [])
    if not feature_names:
        raise PredictionError("The bundle does not contain the training feature list.")

    frame = standardize_column_names(df)

    target = bundle.get("target_name")
    drop = {c for c in (target, *config.DEFAULT_DROP_COLUMNS) if c}
    present = [c for c in drop if c in frame.columns]
    if present:
        frame = frame.drop(columns=present)

    frame = _apply_encoders(frame, bundle.get("encoders") or {})

    for column in feature_names:
        if column not in frame.columns:
            frame[column] = 0.0

    frame = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Impute with the values seen during training when available, else 0.
    medians = {
        column: float(frame[column].median()) if frame[column].notna().any() else 0.0
        for column in feature_names
    }
    frame = frame.fillna(value=medians)

    scaler = bundle.get("scaler")
    matrix = frame.values.astype(float)
    if scaler is not None:
        try:
            matrix = scaler.transform(matrix)
        except Exception as exc:
            raise PredictionError(f"Feature scaling failed: {exc}") from exc
    return matrix


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict_with_bundle(
    df: pd.DataFrame,
    bundle: dict[str, Any] | str | Path | None = None,
    include_probabilities: bool = True,
    prediction_column: str = "prediction",
    label_column: str = "predicted_label",
) -> pd.DataFrame:
    """
    Score ``df`` with a saved bundle and return the input frame plus predictions.

    The bundle is either passed directly or loaded from ``bundle_path``
    (defaults to ``models/automl_bundle.pkl``).
    """
    if bundle is None or isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle)

    model = bundle.get("model")
    if model is None:
        raise PredictionError("The bundle does not contain a fitted model.")

    matrix = encode_for_bundle(df, bundle)
    result = standardize_column_names(df).copy()

    # ``np.ravel`` normalises estimators that return a column vector (CatBoost).
    predictions = np.ravel(model.predict(matrix))
    result[prediction_column] = predictions

    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        try:
            result[label_column] = label_encoder.inverse_transform(
                np.asarray(predictions).astype(int)
            )
        except Exception:
            result[label_column] = predictions
    else:
        result[label_column] = predictions

    if include_probabilities and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(matrix)
            classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
            labels = bundle.get("class_labels") or [str(c) for c in classes]
            for index, class_label in enumerate(classes):
                name = labels[index] if index < len(labels) else class_label
                result[f"probability_{name}"] = np.round(probabilities[:, index], 6)
            result["confidence"] = np.round(probabilities.max(axis=1), 6)
        except Exception:
            pass

    return result


def predict_dataframe(
    model: Any,
    df: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    scaler: Any | None = None,
    encoders: dict[str, Any] | None = None,
) -> np.ndarray:
    """Lower-level helper: score ``df`` with an already loaded ``model``."""
    frame = standardize_column_names(df)
    if encoders:
        frame = _apply_encoders(frame, encoders)
    if feature_names is not None:
        for column in feature_names:
            if column not in frame.columns:
                frame[column] = 0.0
        frame = frame[list(feature_names)]
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = frame.values.astype(float)
    if scaler is not None:
        matrix = scaler.transform(matrix)
    return np.ravel(model.predict(matrix))


# --------------------------------------------------------------------------- #
# Metadata / comparison utilities
# --------------------------------------------------------------------------- #
def model_metadata(bundle: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Return a readable summary of a saved bundle (model name, metrics, features)."""
    if bundle is None or isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle)

    return {
        "model_name": bundle.get("model_name", "unknown"),
        "target": bundle.get("target_name"),
        "classes": bundle.get("class_labels", []),
        "n_features": len(bundle.get("feature_names") or []),
        "feature_names": list(bundle.get("feature_names") or []),
        "metrics": round_metrics(bundle.get("metrics") or {}),
        "version": bundle.get("version", "1.0.0"),
    }


def evaluate_predictions(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    """Accuracy / Precision / Recall / F1 of a set of predictions (used by the UI)."""
    return round_metrics(
        {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    )


def compare_saved_models(
    bundles: dict[str, dict[str, Any] | str | Path],
    df: pd.DataFrame,
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Score ``df`` with several bundles and report the metrics side by side.

    When ``df`` contains the ground-truth ``target_column`` the accuracy and
    F1-score are computed as well, which is how a freshly trained model can be
    benchmarked against a previously saved one.
    """
    rows: dict[str, dict[str, float]] = {}

    for name, bundle in bundles.items():
        loaded = load_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
        scored = predict_with_bundle(df, bundle=loaded)

        row: dict[str, float] = {}
        if target_column:
            standardised = standardize_column_names(df)
            # Accept both the raw header ("Status") and its snake_case form.
            reference = standardised[resolve_column(standardised, target_column)].astype(str)
            predicted = scored["predicted_label"].astype(str)
            row.update(evaluate_predictions(reference, predicted))
        rows[name] = row

    frame = pd.DataFrame.from_dict(rows, orient="index")
    return frame.sort_values(by=frame.columns[0], ascending=False) if not frame.empty else frame
