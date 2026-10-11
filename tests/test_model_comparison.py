"""
Unit tests for ``src.model_comparison`` - the GridSearchCV comparison loop of
Module 3 plus its reporting helpers.
"""

from __future__ import annotations

import unittest

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from tests.helpers import make_classification_frame
from src.model_comparison import (
    PARAM_GRIDS,
    best_model_per_metric,
    compare_best_model,
    comparison_table,
    evaluate_models_with_grid_search,
    metric_series,
    overall_comparison,
    score_all_models,
)
from src.model_training import get_candidate_models
from src.preprocessing import prepare_dataset


class ComparisonFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = prepare_dataset(
            make_classification_frame(rows=180), "status", test_size=0.25, random_state=13
        )
        cls.models = get_candidate_models(selected=["Decision Tree", "Extra Trees"])


class EvaluateModelsTests(ComparisonFixture):
    def test_untuned_models_are_scored_on_the_training_split(self) -> None:
        fitted, results = evaluate_models_with_grid_search(
            self.models,
            param_grid={},
            x_train_scaled=self.prepared.x_train,
            y_train=self.prepared.y_train,
            x_test_scaled=self.prepared.x_test,
            y_test=self.prepared.y_test,
        )
        self.assertEqual(sorted(fitted), ["Decision Tree", "Extra Trees"])
        for score in results.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_grid_search_path_is_used_when_a_grid_is_given(self) -> None:
        fitted, results = evaluate_models_with_grid_search(
            get_candidate_models(selected=["Decision Tree"]),
            param_grid={"Decision Tree": {"max_depth": [2, 4]}},
            x_train_scaled=self.prepared.x_train,
            y_train=self.prepared.y_train,
            cv=3,
        )
        self.assertEqual(sorted(fitted), ["Decision Tree"])
        self.assertGreaterEqual(results["Decision Tree"], 0.0)

    def test_alias_argument_names_are_accepted(self) -> None:
        fitted, _ = evaluate_models_with_grid_search(
            self.models,
            param_grid={},
            x_train=self.prepared.x_train,
            y_train_values=self.prepared.y_train,
            cv=3,
        )
        self.assertEqual(len(fitted), 2)

    def test_missing_training_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_models_with_grid_search(self.models, param_grid={})

    def test_shared_parameter_grids_are_exposed(self) -> None:
        self.assertIn("Random Forest", PARAM_GRIDS)
        self.assertIn("n_estimators", PARAM_GRIDS["Random Forest"])


class ReportingTests(ComparisonFixture):
    def test_compare_best_model_returns_test_accuracy(self) -> None:
        fitted = {
            "weak": DecisionTreeClassifier(max_depth=1, random_state=0).fit(
                self.prepared.x_train, self.prepared.y_train
            ),
            "strong": DecisionTreeClassifier(random_state=0).fit(
                self.prepared.x_train, self.prepared.y_train
            ),
        }
        name, accuracy = compare_best_model(
            fitted,
            {"weak": 0.1, "strong": 0.9},
            self.prepared.x_test,
            self.prepared.y_test,
        )
        self.assertEqual(name, "strong")
        self.assertGreaterEqual(accuracy, 0.0)

    def test_compare_best_model_without_results_raises(self) -> None:
        with self.assertRaises(ValueError):
            compare_best_model({}, {}, self.prepared.x_test, self.prepared.y_test)

    def test_score_all_models_and_comparison_table(self) -> None:
        fitted, _ = evaluate_models_with_grid_search(
            self.models,
            param_grid={},
            x_train_scaled=self.prepared.x_train,
            y_train=self.prepared.y_train,
        )
        performance = score_all_models(fitted, self.prepared.x_test, self.prepared.y_test)
        table = comparison_table(performance, sort_by="accuracy")

        self.assertEqual(set(table.index), {"Decision Tree", "Extra Trees"})
        self.assertIn("accuracy", table.columns)
        self.assertGreaterEqual(table["accuracy"].iloc[0], table["accuracy"].iloc[-1])

    def test_metric_series_is_sorted_descending(self) -> None:
        performance = {
            "A": {"accuracy": 0.4},
            "B": {"accuracy": 0.8},
        }
        series = metric_series(performance, "accuracy")
        self.assertEqual(series.index[0], "B")
        self.assertEqual(series.name, "Accuracy")

    def test_metric_series_rejects_unknown_metrics(self) -> None:
        with self.assertRaises(KeyError):
            metric_series({"A": {"accuracy": 0.5}}, "not_a_metric")

    def test_best_model_per_metric(self) -> None:
        performance = {
            "A": {"accuracy": 0.9, "f1_score": 0.5, "auc": float("nan")},
            "B": {"accuracy": 0.8, "f1_score": 0.7, "auc": 0.95},
        }
        winners = best_model_per_metric(performance)
        self.assertEqual(winners["accuracy"]["model"], "A")
        self.assertEqual(winners["f1_score"]["model"], "B")
        self.assertEqual(winners["auc"]["model"], "B")

    def test_overall_comparison_wrapper(self) -> None:
        name, accuracy, table, fitted = overall_comparison(
            self.models,
            self.prepared.x_train,
            self.prepared.y_train,
            self.prepared.x_test,
            self.prepared.y_test,
            param_grid={},
        )
        self.assertIn(name, ("Decision Tree", "Extra Trees"))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertIsInstance(table, pd.DataFrame)
        self.assertEqual(set(fitted), {"Decision Tree", "Extra Trees"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
