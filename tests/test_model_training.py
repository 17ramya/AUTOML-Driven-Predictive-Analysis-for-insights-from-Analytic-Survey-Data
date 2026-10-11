"""
Unit tests for ``src.model_training`` - the candidate registry, metric
computation, model selection, tuning and joblib persistence.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from tests.helpers import make_binary_frame, make_classification_frame
from src import config
from src.model_training import (
    PERFORMANCE_METRIC_KEYS,
    available_model_names,
    calculate_metrics,
    compute_overall_score,
    cross_validate_model,
    get_candidate_models,
    load_bundle,
    load_model,
    run_training_pipeline,
    save_bundle,
    save_model,
    select_best_model,
    train_and_evaluate_models,
    tune_model,
    performance_dataframe,
)
from src.preprocessing import prepare_dataset

OPTIONAL_MODELS = ("XGBoost", "CatBoost", "LightGBM")


class CandidateModelTests(unittest.TestCase):
    def test_registry_contains_the_thesis_models(self) -> None:
        names = available_model_names(include_optional=False)
        for expected in (
            "Random Forest",
            "Gradient Boosting",
            "Extra Trees",
            "Logistic Regression",
            "Ridge Classifier",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "Gaussian Naive Bayes",
            "AdaBoost",
            "Linear Discriminant Analysis",
            "Quadratic Discriminant Analysis",
            "Decision Tree",
        ):
            with self.subTest(model=expected):
                self.assertIn(expected, names)

    def test_optional_boosters_can_be_excluded(self) -> None:
        without = available_model_names(include_optional=False)
        for name in OPTIONAL_MODELS:
            self.assertNotIn(name, without)

    def test_selected_filter_is_case_insensitive(self) -> None:
        selected = get_candidate_models(selected=["extra trees", "K-NEAREST NEIGHBORS "])
        self.assertEqual([name for name, _ in selected], ["Extra Trees", "K-Nearest Neighbors"])

    def test_every_candidate_is_cloneable(self) -> None:
        from sklearn.base import clone

        for name, estimator in get_candidate_models():
            with self.subTest(model=name):
                self.assertIsNotNone(clone(estimator))


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = make_classification_frame(rows=200)
        cls.prepared = prepare_dataset(cls.frame, "status", test_size=0.25, random_state=42)
        cls.model = DecisionTreeClassifier(random_state=42)
        cls.model.fit(cls.prepared.x_train, cls.prepared.y_train)

    def test_calculate_metrics_returns_every_reported_metric(self) -> None:
        metrics = calculate_metrics(
            self.model, self.prepared.x_test, self.prepared.y_test, training_time=0.5
        )
        for key in PERFORMANCE_METRIC_KEYS:
            with self.subTest(metric=key):
                self.assertIn(key, metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["training_time"], 0.5)

    def test_auc_is_available_for_decision_function_only_models(self) -> None:
        # Ridge Classifier exposes only `decision_function`; the softmax fallback
        # must still produce a usable one-vs-rest AUC.
        ridge = get_candidate_models(selected=["Ridge Classifier"])[0][1]
        ridge.fit(self.prepared.x_train, self.prepared.y_train)
        metrics = calculate_metrics(ridge, self.prepared.x_test, self.prepared.y_test)
        self.assertFalse(np.isnan(metrics["auc"]))
        self.assertGreater(metrics["auc"], 0.5)

    def test_overall_score_is_a_weighted_average(self) -> None:
        metrics = {"accuracy": 1.0, "f1_score": 0.5}
        self.assertAlmostEqual(compute_overall_score(metrics, weights={"accuracy": 1.0}), 1.0)
        self.assertAlmostEqual(
            compute_overall_score(metrics, weights={"accuracy": 1.0, "f1_score": 1.0}), 0.75
        )

    def test_missing_metrics_contribute_zero(self) -> None:
        self.assertEqual(compute_overall_score({"accuracy": float("nan")}, {"accuracy": 1.0}), 0.0)
        self.assertEqual(compute_overall_score({}, config.SCORE_WEIGHTS), 0.0)


class SelectionTests(unittest.TestCase):
    def test_select_best_model_prefers_accuracy(self) -> None:
        performance = {
            "A": {"accuracy": 0.9, "f1_score": 0.9, "auc": 0.9},
            "B": {"accuracy": 0.95, "f1_score": 0.5, "auc": 0.5},
        }
        self.assertEqual(select_best_model(performance)[0], "B")

    def test_tie_breaking_uses_f1_then_auc(self) -> None:
        performance = {
            "A": {"accuracy": 0.9, "f1_score": 0.91, "auc": 0.5},
            "B": {"accuracy": 0.9, "f1_score": 0.80, "auc": 0.99},
            "C": {"accuracy": 0.9, "f1_score": float("nan"), "auc": 1.0},
        }
        self.assertEqual(select_best_model(performance)[0], "A")

    def test_empty_performance_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_best_model({})

    def test_performance_dataframe_has_the_expected_columns(self) -> None:
        performance = {
            "A": {key: 0.5 for key in PERFORMANCE_METRIC_KEYS},
            "B": {key: 0.75 for key in PERFORMANCE_METRIC_KEYS},
        }
        frame = performance_dataframe(performance)
        self.assertEqual(list(frame.columns), list(config.METRIC_ORDER))
        self.assertEqual(frame.index[0], "B")


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = prepare_dataset(
            make_classification_frame(rows=120), "status", test_size=0.25, random_state=1
        )

    def test_tune_model_returns_best_estimator_and_info(self) -> None:
        best, info = tune_model(
            DecisionTreeClassifier(random_state=0),
            {"max_depth": [2, 4]},
            self.prepared.x_train,
            self.prepared.y_train,
            cv=3,
        )
        self.assertIn(info["best_params"]["max_depth"], (2, 4))
        self.assertGreaterEqual(info["best_cv_score"], 0.0)
        self.assertTrue(hasattr(best, "predict"))

    def test_train_and_evaluate_models_scores_every_candidate(self) -> None:
        models = get_candidate_models(selected=["Decision Tree", "Extra Trees"])
        trained, performance, tuning_info = train_and_evaluate_models(
            models,
            self.prepared.x_train,
            self.prepared.y_train,
            self.prepared.x_test,
            self.prepared.y_test,
            tune=False,
            verbose=False,
        )
        self.assertEqual(sorted(trained), ["Decision Tree", "Extra Trees"])
        self.assertEqual(sorted(performance), ["Decision Tree", "Extra Trees"])
        self.assertEqual(tuning_info, {})
        for metrics in performance.values():
            self.assertIn("accuracy", metrics)
            self.assertIn("score", metrics)

    def test_tuning_populates_the_tuning_info(self) -> None:
        models = get_candidate_models(selected=["Decision Tree"])
        _trained, _performance, tuning_info = train_and_evaluate_models(
            models,
            self.prepared.x_train,
            self.prepared.y_train,
            self.prepared.x_test,
            self.prepared.y_test,
            param_grids={"Decision Tree": {"max_depth": [2, 4]}},
            tune=True,
            cv=3,
            verbose=False,
        )
        self.assertIn("Decision Tree", tuning_info)

    def test_a_failing_estimator_is_skipped(self) -> None:
        broken = LogisticRegression(penalty="l1", solver="lbfgs", max_iter=50)
        models = [("Broken", broken), ("Extra Trees", DecisionTreeClassifier(random_state=0))]
        trained, performance, _ = train_and_evaluate_models(
            models,
            self.prepared.x_train,
            self.prepared.y_train,
            self.prepared.x_test,
            self.prepared.y_test,
            verbose=False,
        )
        self.assertNotIn("Broken", trained)
        self.assertIn("Extra Trees", performance)

    def test_cross_validation_summary(self) -> None:
        summary = cross_validate_model(
            DecisionTreeClassifier(random_state=0),
            self.prepared.x_train,
            self.prepared.y_train,
            cv=3,
        )
        self.assertEqual(summary["cv"], 3)
        self.assertEqual(len(summary["scores"]), 3)
        self.assertGreaterEqual(summary["mean"], 0.0)
        self.assertLessEqual(summary["mean"], 1.0)


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = prepare_dataset(
            make_classification_frame(rows=120), "status", test_size=0.25, random_state=5
        )
        cls.model = DecisionTreeClassifier(random_state=5)
        cls.model.fit(cls.prepared.x_train, cls.prepared.y_train)

    def test_save_and_load_model_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(self.model, Path(tmp) / "tree.pkl")
            self.assertTrue(path.exists())

            restored = load_model(path)
            np.testing.assert_array_equal(
                restored.predict(self.prepared.x_test), self.model.predict(self.prepared.x_test)
            )

    def test_default_model_path_follows_the_thesis_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = config.MODELS_DIR
            config.MODELS_DIR = Path(tmp)
            try:
                path = save_model(self.model, name="Extra Trees")
            finally:
                config.MODELS_DIR = previous
            self.assertEqual(path.name, "Extra_Trees_final_model.pkl")

    def test_bundle_round_trip_contains_the_preprocessing_artefacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(
                Path(tmp) / "bundle.pkl",
                self.model,
                "Decision Tree",
                self.prepared,
                metrics={"accuracy": 0.9},
            )
            bundle = load_bundle(path)

        for key in (
            "model",
            "model_name",
            "encoders",
            "scaler",
            "label_encoder",
            "feature_names",
            "target_name",
            "class_labels",
            "metrics",
            "version",
        ):
            with self.subTest(key=key):
                self.assertIn(key, bundle)
        self.assertEqual(bundle["model_name"], "Decision Tree")
        self.assertEqual(bundle["target_name"], "status")
        self.assertEqual(bundle["feature_names"], self.prepared.feature_names)
        self.assertEqual(bundle["metrics"]["accuracy"], 0.9)

    def test_load_bundle_rejects_a_foreign_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_a_bundle.pkl"
            save_model(self.model, path)
            with self.assertRaises(ValueError):
                load_bundle(path)


class RunTrainingPipelineTests(unittest.TestCase):
    def test_end_to_end_pipeline_persists_model_and_bundle(self) -> None:
        frame = make_classification_frame(rows=240)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_training_pipeline(
                frame,
                "status",
                selected_models=["Extra Trees", "Decision Tree", "Logistic Regression"],
                test_size=0.25,
                save=True,
                generate_eda=False,
                model_dir=Path(tmp),
                verbose=False,
            )

            self.assertTrue(result.model_path is not None and result.model_path.exists())
            self.assertTrue(result.bundle_path is not None and result.bundle_path.exists())
            bundle = load_bundle(result.bundle_path)

        self.assertIn(result.best_model_name, result.performance)
        self.assertGreaterEqual(result.best_metrics["accuracy"], 0.6)
        self.assertGreater(result.best_metrics["auc"], 0.6)
        self.assertEqual(result.prepared.n_features, 7)
        self.assertNotIn("order_id", result.prepared.feature_names)
        self.assertEqual(len(result.performance), 3)
        self.assertEqual(bundle["model_name"], result.best_model_name)

        summary = result.to_dict()
        self.assertEqual(summary["models_evaluated"], 3)
        self.assertEqual(summary["dataset"]["target"], "status")
        self.assertIn("cv_summary", summary)
        self.assertEqual(list(result.table.columns), list(config.METRIC_ORDER))

    def test_pipeline_without_persistence_and_binary_target(self) -> None:
        frame = make_binary_frame(rows=160)
        result = run_training_pipeline(
            frame,
            "outcome",
            selected_models=["Decision Tree"],
            test_size=0.25,
            save=False,
            generate_eda=False,
            verbose=False,
        )
        self.assertEqual(result.best_model_name, "Decision Tree")
        self.assertIsNone(result.model_path)
        self.assertIsNone(result.bundle_path)
        self.assertGreaterEqual(result.best_metrics["accuracy"], 0.6)

    def test_unknown_model_selection_raises(self) -> None:
        frame = make_binary_frame(rows=80)
        with self.assertRaises(ValueError):
            run_training_pipeline(
                frame,
                "outcome",
                selected_models=["No Such Model"],
                save=False,
                generate_eda=False,
                verbose=False,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
