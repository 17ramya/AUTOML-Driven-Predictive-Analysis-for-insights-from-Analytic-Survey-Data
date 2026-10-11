"""
MODULE 2 - Model Training and Selection
=====================================
Thesis reference: Sections 3.3.3 / 3.3.4 (pseudo code) and 4.4 / 4.4.1 / 4.4.2.

Pipeline implemented here
-------------------------
1. Build the candidate classifier list (the auto-detected model "zoo").
2. Optionally tune XGBoost and Random Forest with ``GridSearchCV``.
3. Train every candidate on the scaled training split.
4. Score every candidate with Accuracy, Precision, Recall, F1-Score, MCC,
   Kappa, AUC and the wall-clock training time.
5. Rank the candidates (Accuracy -> F1 -> AUC) and pick the best model.
6. Retrain the winner on the **full** dataset and serialise it with ``joblib``
   so that it can be reused / deployed.
7. Report a 5-fold cross-validation score as a stability check
   (the thesis performs this on Logistic Regression).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from . import config
from .preprocessing import PreparedDataset, prepare_dataset
from .utils import ensure_directory, round_metrics, safe_filename

__all__ = [
    "TrainingResult",
    "get_candidate_models",
    "available_model_names",
    "calculate_metrics",
    "compute_overall_score",
    "tune_model",
    "train_and_evaluate_models",
    "select_best_model",
    "performance_dataframe",
    "cross_validate_model",
    "save_model",
    "save_bundle",
    "load_model",
    "run_training_pipeline",
    "PERFORMANCE_METRIC_KEYS",
]

PERFORMANCE_METRIC_KEYS: tuple[str, ...] = (
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


# --------------------------------------------------------------------------- #
# Optional gradient-boosting libraries (imported lazily / defensively)
# --------------------------------------------------------------------------- #
def _make_xgboost(random_state: int) -> Any | None:
    try:  # pragma: no cover - depends on the environment
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    except Exception:
        return None


def _make_catboost(random_state: int) -> Any | None:
    try:  # pragma: no cover
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            learning_rate=0.1, iterations=300, random_seed=random_state, verbose=0
        )
    except Exception:
        return None


def _make_lightgbm(random_state: int) -> Any | None:
    try:  # pragma: no cover
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=random_state, verbose=-1)
    except Exception:
        return None


def get_candidate_models(
    random_state: int = config.RANDOM_STATE,
    include_optional: bool = True,
    selected: Iterable[str] | None = None,
) -> list[tuple[str, BaseEstimator]]:
    """
    Return the list of ``(name, estimator)`` candidates to evaluate.

    The registry mirrors Section 5.2.1 of the report: Random Forest, Gradient
    Boosting, Extra Trees, LightGBM, Logistic Regression, AdaBoost, K-Nearest
    Neighbours, LDA, QDA, Ridge, SVM - plus XGBoost and CatBoost from the
    Section 4.4.2 listing, and Gaussian Naive Bayes / Decision Tree.

    Estimators whose optional dependency is missing are silently skipped, so
    the pipeline keeps working on a minimal installation.
    """
    models: list[tuple[str, BaseEstimator]] = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=100, random_state=random_state)),
        ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=random_state)),
        ("Ridge Classifier", RidgeClassifier(random_state=random_state)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Support Vector Machine", SVC(probability=True, random_state=random_state)),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=random_state)),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
    ]

    if include_optional:
        optional = [
            ("XGBoost", _make_xgboost),
            ("CatBoost", _make_catboost),
            ("LightGBM", _make_lightgbm),
        ]
        for name, factory in optional:
            estimator = factory(random_state)
            if estimator is not None:
                models.append((name, estimator))

    if selected is not None:
        wanted = {str(s).strip().lower() for s in selected}
        models = [(n, m) for n, m in models if n.lower() in wanted]

    return models


def available_model_names(include_optional: bool = True) -> list[str]:
    """Names of every model that can be instantiated in this environment."""
    return [name for name, _ in get_candidate_models(include_optional=include_optional)]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax so decision-function outputs can be used as probabilities."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _safe_auc(estimator: BaseEstimator, x_test: np.ndarray, y_test: pd.Series) -> float:
    """
    AUC (one-vs-rest for multi-class). Returns ``nan`` when it cannot be computed.

    ``predict_proba`` is preferred; models that only expose ``decision_function``
    (e.g. Ridge Classifier, SVM without probabilities) have their scores passed
    through a softmax so that ``roc_auc_score(..., multi_class="ovr")`` accepts
    them.
    """
    try:
        classes = np.unique(y_test)
        if classes.size < 2:
            return float("nan")

        if hasattr(estimator, "predict_proba"):
            scores = np.asarray(estimator.predict_proba(x_test), dtype=float)
        elif hasattr(estimator, "decision_function"):
            raw = np.asarray(estimator.decision_function(x_test), dtype=float)
            if raw.ndim == 1:
                scores = raw
            elif raw.shape[1] == 2:
                scores = raw[:, 1]
            else:
                scores = _softmax(raw)
        else:
            return float("nan")

        if scores.ndim == 1:
            return float(roc_auc_score(y_test, scores))

        return float(roc_auc_score(y_test, scores, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def calculate_metrics(
    estimator: BaseEstimator,
    x_test: np.ndarray,
    y_test: pd.Series,
    training_time: float = 0.0,
) -> dict[str, float]:
    """
    Compute the full metric set reported in Section 5.2.2 of the thesis.

    Accuracy, Precision, Recall, F1-Score, MCC, Kappa, AUC, training time and
    the composite overall score are returned as plain floats.
    """
    # ``np.ravel`` keeps exotic estimators (e.g. CatBoost, which can return a
    # column vector) compatible with the scikit-learn metric functions.
    predictions = np.ravel(estimator.predict(x_test))

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, predictions)),
        "kappa": float(cohen_kappa_score(y_test, predictions)),
        "auc": _safe_auc(estimator, x_test, y_test),
        "training_time": float(training_time),
    }
    metrics["score"] = compute_overall_score(metrics)
    return metrics


def compute_overall_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """
    Weighted composite score used by the bar chart of Figure 5.12.

    Unavailable metrics (``nan``) contribute 0 so that a model is never
    rewarded for a missing value.
    """
    active_weights = weights or config.SCORE_WEIGHTS
    total_weight = sum(active_weights.values()) or 1.0
    score = 0.0
    for key, weight in active_weights.items():
        value = metrics.get(key, 0.0)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = 0.0
        score += float(value) * (weight / total_weight)
    return round(float(score), 6)


# --------------------------------------------------------------------------- #
# Hyper-parameter tuning (Section 4.4.1 - GridSearchCV on XGBoost / RandomForest)
# --------------------------------------------------------------------------- #
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Random Forest": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    },
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "Decision Tree": {"max_depth": [3, 5, 10, None], "criterion": ["gini", "entropy"]},
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
    "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
}


def tune_model(
    estimator: BaseEstimator,
    param_grid: dict[str, list[Any]],
    x_train: np.ndarray,
    y_train: pd.Series,
    cv: int = config.CV_FOLDS,
    n_jobs: int = config.N_JOBS,
    verbose: int = 0,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """
    Run ``GridSearchCV`` for ``estimator`` and return ``(best_estimator, info)``.

    ``info`` carries the winning parameter combination and the best mean
    cross-validation score, exactly as printed in the report.
    """
    search = GridSearchCV(
        estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose
    )
    search.fit(x_train, y_train)
    info = {
        "best_params": dict(search.best_params_),
        "best_cv_score": float(search.best_score_),
    }
    return search.best_estimator_, info


# --------------------------------------------------------------------------- #
# Training + evaluation
# --------------------------------------------------------------------------- #
def train_and_evaluate_models(
    models: Sequence[tuple[str, BaseEstimator]],
    x_train: np.ndarray,
    y_train: pd.Series,
    x_test: np.ndarray,
    y_test: pd.Series,
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
    tune: bool = False,
    cv: int = config.CV_FOLDS,
    verbose: bool = True,
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    """
    Train and score every candidate model.

    Returns
    -------
    (trained_models, performance, tuning_info)
        * ``trained_models`` - name -> fitted estimator
        * ``performance``    - name -> full metric dictionary
        * ``tuning_info``    - name -> GridSearchCV summary (empty when ``tune=False``)
    """
    grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS
    trained_models: dict[str, BaseEstimator] = {}
    performance: dict[str, dict[str, float]] = {}
    tuning_info: dict[str, dict[str, Any]] = {}

    for name, estimator in models:
        try:
            model = clone(estimator)

            if tune and name in grids:
                model, info = tune_model(model, grids[name], x_train, y_train, cv=cv)
                tuning_info[name] = info
                if verbose:
                    print(f"  [tuned] {name}: cv={info['best_cv_score']:.4f} params={info['best_params']}")

            started = time.perf_counter()
            model.fit(x_train, y_train)
            elapsed = time.perf_counter() - started

            metrics = calculate_metrics(model, x_test, y_test, training_time=round(elapsed, 4))
            trained_models[name] = model
            performance[name] = metrics

            if verbose:
                print(
                    f"  {name:<32} accuracy={metrics['accuracy']:.4f} "
                    f"f1={metrics['f1_score']:.4f} auc={metrics['auc']:.4f} "
                    f"time={metrics['training_time']:.3f}s"
                )
        except Exception as exc:  # keep going even when a single estimator fails
            if verbose:
                print(f"  [skipped] {name}: {exc}")

    return trained_models, performance, tuning_info


def select_best_model(
    performance: dict[str, dict[str, float]],
    primary: str = "accuracy",
    tie_breakers: Sequence[str] = ("f1_score", "auc"),
) -> tuple[str, dict[str, float]]:
    """
    Rank the candidates by ``primary`` with deterministic tie-breaking.

    Matches the hierarchical ordering of the report: Accuracy, then F1-Score,
    then AUC (Section 4.4).
    """
    if not performance:
        raise ValueError("No model performance scores available to rank.")

    def sort_key(item: tuple[str, dict[str, float]]) -> tuple[float, ...]:
        _name, metrics = item
        values: list[float] = []
        for key in (primary, *tie_breakers):
            value = metrics.get(key, float("nan"))
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = -1.0
            values.append(float(value))
        return tuple(values)

    best_name, best_metrics = max(performance.items(), key=sort_key)
    return best_name, best_metrics


def performance_dataframe(performance: dict[str, dict[str, float]], sort_by: str = "accuracy") -> pd.DataFrame:
    """Convert the performance dictionary into a sorted, display-ready DataFrame."""
    frame = pd.DataFrame.from_dict(performance, orient="index")
    for column in config.METRIC_ORDER:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[list(config.METRIC_ORDER)]
    if sort_by in frame.columns:
        frame = frame.sort_values(by=sort_by, ascending=False)
    return frame.round(4)


def cross_validate_model(
    estimator: BaseEstimator,
    x: np.ndarray,
    y: pd.Series,
    cv: int = config.CV_FOLDS_CROSS_VAL,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """k-fold cross-validation stability check (performed on Logistic Regression in the thesis)."""
    scores = cross_val_score(estimator, x, y, cv=cv, scoring=scoring)
    return {
        "scores": [float(s) for s in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "cv": int(cv),
        "scoring": scoring,
    }


# --------------------------------------------------------------------------- #
# Persistence (joblib) - "SAVEFILE(joblib.dump(best_model, 'best_model.pkl'))"
# --------------------------------------------------------------------------- #
def save_model(model: BaseEstimator, path: str | Path | None = None, name: str | None = None) -> Path:
    """
    Serialise a single estimator with ``joblib``.

    When ``path`` is omitted the file is written to ``models/<Name>_final_model.pkl``
    (the naming convention used in the report).
    """
    if path is None:
        stem = safe_filename(name) if name else "best"
        path = config.MODELS_DIR / f"{stem}_final_model.pkl"
    target = Path(path)
    ensure_directory(target.parent)
    joblib.dump(model, target)
    return target


def load_model(path: str | Path) -> BaseEstimator:
    """Load a previously serialised estimator."""
    return joblib.load(Path(path))


def save_bundle(
    path: str | Path | None,
    model: BaseEstimator,
    model_name: str,
    prepared: PreparedDataset,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Serialise the model **together with** its preprocessing artefacts.

    A bare estimator is not enough to serve predictions: the incoming raw frame
    still needs the same encoders and the same ``StandardScaler``.  The bundle
    therefore stores ``model``, ``encoders``, ``scaler``, ``target_name``,
    ``feature_names`` and the metric dictionary in one ``.pkl`` file.
    """
    target = Path(path) if path is not None else config.BUNDLE_PATH
    ensure_directory(target.parent)

    bundle = {
        "model": model,
        "model_name": model_name,
        "encoders": dict(prepared.encoders),
        "scaler": prepared.scaler,
        "label_encoder": prepared.label_encoder,
        "feature_names": list(prepared.feature_names),
        "target_name": prepared.target_name,
        "class_labels": prepared.class_labels,
        "metrics": dict(metrics or {}),
        "metadata": dict(extra or {}),
        "version": "1.0.0",
    }
    joblib.dump(bundle, target)
    return target


def load_bundle(path: str | Path | None = None) -> dict[str, Any]:
    """Load a bundle written by :func:`save_bundle`."""
    target = Path(path) if path is not None else config.BUNDLE_PATH
    bundle = joblib.load(target)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{target} is not a valid AutoML bundle.")
    return bundle


@dataclass
class TrainingResult:
    """Everything produced by :func:`run_training_pipeline`."""

    best_model_name: str
    best_model: BaseEstimator
    performance: dict[str, dict[str, float]]
    prepared: PreparedDataset
    table: pd.DataFrame
    model_path: Path | None = None
    bundle_path: Path | None = None
    cv_summary: dict[str, Any] = field(default_factory=dict)
    tuning_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    eda_path: Path | None = None

    @property
    def best_metrics(self) -> dict[str, float]:
        return self.performance.get(self.best_model_name, {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_model_name": self.best_model_name,
            "best_metrics": round_metrics(self.best_metrics),
            "model_path": str(self.model_path) if self.model_path else None,
            "bundle_path": str(self.bundle_path) if self.bundle_path else None,
            "cv_summary": self.cv_summary,
            "dataset": self.prepared.summary(),
            "models_evaluated": len(self.performance),
        }


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def run_training_pipeline(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Sequence[str] = config.DEFAULT_DROP_COLUMNS,
    selected_models: Iterable[str] | None = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    tune: bool = True,
    include_optional: bool = True,
    cv: int = config.CV_FOLDS,
    save: bool = True,
    generate_eda: bool = True,
    eda_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """
    Run the complete Module 2 workflow (preprocess -> train -> rank -> persist).

    This is the single function the Streamlit app and the CLI both call, which
    keeps the interactive and batch workflows perfectly in sync.
    """
    prepared = prepare_dataset(
        df,
        target_column=target_column,
        drop_columns=drop_columns,
        test_size=test_size,
        random_state=random_state,
    )

    eda_file: Path | None = None
    if generate_eda:
        from .eda import generate_eda_report  # local import avoids a heavy import cost

        eda_file = generate_eda_report(df, output_path=eda_path)

    models = get_candidate_models(
        random_state=random_state, include_optional=include_optional, selected=selected_models
    )
    if not models:
        raise ValueError("No candidate models selected. Check the `selected_models` argument.")

    if verbose:
        print(
            f"Training {len(models)} model(s) on {prepared.x_train.shape[0]} rows "
            f"x {prepared.n_features} features (target='{prepared.target_name}', "
            f"{prepared.n_classes} classes) ..."
        )

    trained_models, performance, tuning_info = train_and_evaluate_models(
        models,
        prepared.x_train,
        prepared.y_train,
        prepared.x_test,
        prepared.y_test,
        tune=tune,
        cv=cv,
        verbose=verbose,
    )

    if not performance:
        raise RuntimeError("Every candidate model failed to train. Check the dataset.")

    best_name, _ = select_best_model(performance)
    best_model = trained_models[best_name]

    # Retrain the winner on the *entire* dataset ("best_model.fit(X, y)").
    best_model.fit(prepared.x_full, prepared.y_full)

    cv_summary = cross_validate_model(
        LogisticRegression(max_iter=1000, random_state=random_state),
        prepared.x_train,
        prepared.y_train,
        cv=config.CV_FOLDS_CROSS_VAL,
    )
    if verbose:
        print(f"Best model : {best_name}")
        print(f"5-fold CV  : mean accuracy = {cv_summary['mean']:.4f} (+/- {cv_summary['std']:.4f})")

    model_path: Path | None = None
    bundle_path: Path | None = None
    if save:
        directory = Path(model_dir) if model_dir is not None else config.MODELS_DIR
        ensure_directory(directory)
        model_path = save_model(best_model, path=directory / f"{safe_filename(best_name)}_final_model.pkl")
        bundle_path = save_bundle(
            directory / "automl_bundle.pkl",
            best_model,
            best_name,
            prepared,
            metrics=performance[best_name],
            extra={"tuning_info": tuning_info, "cv_summary": cv_summary},
        )
        if verbose:
            print(f"Saved model : {model_path}")
            print(f"Saved bundle: {bundle_path}")

    return TrainingResult(
        best_model_name=best_name,
        best_model=best_model,
        performance=performance,
        prepared=prepared,
        table=performance_dataframe(performance),
        model_path=model_path,
        bundle_path=bundle_path,
        cv_summary=cv_summary,
        tuning_info=tuning_info,
        eda_path=eda_file,
    )


