"""
MODULE 3 - Model Comparison
=========================
Thesis reference: Sections 3.3.5 and 4.5 / 4.5.1 / 4.5.2, plus the charts of
Figures 5.12 - 5.20.

This module reproduces the exact loop of the thesis code listing::

    def evaluate_models(models, param_grid, X_train_scaled, y_train):
        for model_name, model in models:
            if model_name in param_grid:
                grid_search = GridSearchCV(...); grid_search.fit(...)
                best_model = grid_search.best_estimator_
            else:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_train_scaled, y_train)
            best_models[model_name] = best_model
            best_model_results[model_name] = score
        return best_models, best_model_results

    best_model_name = max(best_model_results, key=best_model_results.get)
    best_model = best_models[best_model_name]
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

In addition it exposes helpers to turn the comparison into the bar charts and
the comparison table presented in the report.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score

from . import config
from .model_training import (
    DEFAULT_PARAM_GRIDS,
    PERFORMANCE_METRIC_KEYS,
    calculate_metrics,
    performance_dataframe,
    select_best_model,
)
from .utils import round_metrics

__all__ = [
    "PARAM_GRIDS",
    "evaluate_models_with_grid_search",
    "compare_best_model",
    "comparison_table",
    "best_model_per_metric",
    "metric_series",
]


# Parameter grids are shared with the training module so that the interactive
# and the batch workflow tune identically.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = DEFAULT_PARAM_GRIDS


def evaluate_models_with_grid_search(
    models: Sequence[tuple[str, BaseEstimator]],
    param_grid: dict[str, dict[str, list[Any]]] | None = None,
    x_train_scaled: np.ndarray | None = None,
    y_train: pd.Series | None = None,
    x_test_scaled: np.ndarray | None = None,
    y_test: pd.Series | None = None,
    *,
    x_train: np.ndarray | None = None,
    y_train_values: pd.Series | None = None,
    x_test: np.ndarray | None = None,
    y_test_values: pd.Series | None = None,
    cv: int = config.CV_FOLDS,
    verbose: bool = False,
) -> tuple[dict[str, BaseEstimator], dict[str, float]]:
    """
    Train every model, tuning the ones that appear in ``param_grid``.

    Accepts both the thesis argument names (``x_train_scaled`` / ``y_train``)
    and the explicit ``x_train`` / ``y_train_values`` aliases.

    Returns
    -------
    (best_models, best_model_results)
        ``best_models``       - name -> fitted estimator (tuned where applicable)
        ``best_model_results``- name -> best cross-validation score, or the
                                training accuracy when no grid was supplied
    """
    grids = param_grid if param_grid is not None else PARAM_GRIDS
    train_x = x_train_scaled if x_train_scaled is not None else x_train
    train_y = y_train if y_train is not None else y_train_values
    test_x = x_test_scaled if x_test_scaled is not None else x_test
    test_y = y_test if y_test is not None else y_test_values

    if train_x is None or train_y is None:
        raise ValueError("Training data must be supplied to evaluate_models_with_grid_search().")

    best_models: dict[str, BaseEstimator] = {}
    best_model_results: dict[str, float] = {}

    for model_name, model in models:
        candidate = clone(model)
        try:
            if model_name in grids:
                from sklearn.model_selection import GridSearchCV

                grid_search = GridSearchCV(
                    estimator=candidate,
                    param_grid=grids[model_name],
                    cv=cv,
                    n_jobs=-1,
                    verbose=1 if verbose else 0,
                )
                grid_search.fit(train_x, train_y)
                best = grid_search.best_estimator_
                score = float(grid_search.best_score_)
                if verbose:
                    print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
                    print(f"Best cross-validation score for {model_name}: {score:.4f}")
            else:
                best = candidate
                best.fit(train_x, train_y)
                score = float(best.score(train_x, train_y))
                if verbose:
                    print(f"Training accuracy for {model_name}: {score:.4f}")
        except Exception as exc:
            if verbose:
                print(f"[skipped] {model_name}: {exc}")
            continue

        best_models[model_name] = best
        best_model_results[model_name] = score

        if test_x is not None and test_y is not None:
            _ = accuracy_score(test_y, best.predict(test_x))  # kept for parity with thesis flow

    return best_models, best_model_results


# --------------------------------------------------------------------------- #
# Best model extraction / validation
# --------------------------------------------------------------------------- #
def compare_best_model(
    best_models: dict[str, BaseEstimator],
    best_model_results: dict[str, float],
    x_test_scaled: np.ndarray,
    y_test: pd.Series,
    verbose: bool = False,
) -> tuple[str, float]:
    """
    Pick the model with the highest (cross-validation) score and validate it on
    the held-out test split.

    Returns ``(best_model_name, test_accuracy)``.
    """
    if not best_model_results:
        raise ValueError("No model results available for comparison.")

    best_model_name = max(best_model_results, key=best_model_results.get)
    best_model = best_models[best_model_name]
    predictions = best_model.predict(x_test_scaled)
    accuracy = float(accuracy_score(y_test, predictions))

    if verbose:
        print(f"Best Model: {best_model_name}")
        print(f"Test accuracy on unseen data: {accuracy:.4f}")

    return best_model_name, accuracy


def score_all_models(
    best_models: dict[str, BaseEstimator],
    x_test: np.ndarray,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """
    Compute the full metric set for every fitted model.

    Feeding the result to :func:`comparison_table` reproduces the comparison
    table discussed in Section 5.2.2 of the report.
    """
    performance: dict[str, dict[str, float]] = {}
    for name, model in best_models.items():
        try:
            started = time.perf_counter()
            model.predict(x_test)
            elapsed = time.perf_counter() - started
            performance[name] = round_metrics(calculate_metrics(model, x_test, y_test, elapsed))
        except Exception:
            continue
    return performance


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def comparison_table(
    performance: dict[str, dict[str, float]],
    sort_by: str = "accuracy",
) -> pd.DataFrame:
    """Sorted comparison table with one row per model (data behind Figures 5.9 / 5.12)."""
    return performance_dataframe(performance, sort_by=sort_by)


def metric_series(performance: dict[str, dict[str, float]], metric: str) -> pd.Series:
    """
    Extract a single metric for every model, sorted descending.

    Used to draw the "Model comparison based on <metric>" bar charts of
    Figures 5.13 - 5.20.
    """
    if metric not in PERFORMANCE_METRIC_KEYS:
        raise KeyError(f"Unknown metric '{metric}'. Expected one of {PERFORMANCE_METRIC_KEYS}.")

    values = {name: metrics.get(metric, float("nan")) for name, metrics in performance.items()}
    series = pd.Series(values, name=config.METRIC_LABELS.get(metric, metric))
    return series.sort_values(ascending=False)


def best_model_per_metric(performance: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    """
    For every performance metric, report the winning model and its value.

    ``{"accuracy": {"model": "Gradient Boosting", "value": 1.0}, ...}``
    """
    winners: dict[str, dict[str, Any]] = {}
    for metric in PERFORMANCE_METRIC_KEYS:
        scores = {name: metrics.get(metric, float("nan")) for name, metrics in performance.items()}
        valid = {
            k: v
            for k, v in scores.items()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        }
        if not valid:
            continue
        winner = max(valid, key=valid.get)
        winners[metric] = {"model": winner, "value": round(float(valid[winner]), 4)}
    return winners


def overall_comparison(
    models: Sequence[tuple[str, BaseEstimator]],
    x_train: np.ndarray,
    y_train: pd.Series,
    x_test: np.ndarray,
    y_test: pd.Series,
    param_grid: dict[str, dict[str, list[Any]]] | None = None,
    cv: int = config.CV_FOLDS,
    verbose: bool = False,
) -> tuple[str, float, pd.DataFrame, dict[str, BaseEstimator]]:
    """
    Convenience wrapper running the whole comparison module end to end.

    Returns ``(best_model_name, test_accuracy, table, fitted_models)``.
    """
    fitted, _results = evaluate_models_with_grid_search(
        models,
        param_grid=param_grid,
        x_train_scaled=x_train,
        y_train=y_train,
        cv=cv,
        verbose=verbose,
    )
    if not fitted:
        raise RuntimeError("No model could be trained during comparison.")

    performance = score_all_models(fitted, x_test, y_test)
    best_name, _ = select_best_model(performance)
    predictions = fitted[best_name].predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    return best_name, accuracy, comparison_table(performance), fitted
