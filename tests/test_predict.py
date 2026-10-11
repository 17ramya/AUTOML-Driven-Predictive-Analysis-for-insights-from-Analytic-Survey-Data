"""
Unit tests for ``src.predict`` - replaying the training preprocessing at
inference time and scoring raw CSV frames.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import make_classification_frame
from src.predict import (
    PredictionError,
    compare_saved_models,
    encode_for_bundle,
    evaluate_predictions,
    model_metadata,
    predict_dataframe,
    predict_with_bundle,
)
from src.model_training import load_bundle, run_training_pipeline
from src.preprocessing import prepare_dataset


class PredictionTestBase(unittest.TestCase):
    """Train a tiny pipeline once and reuse the produced bundle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = make_classification_frame(rows=200)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.result = run_training_pipeline(
            cls.frame,
            "status",
            selected_models=["Extra Trees", "Decision Tree"],
            test_size=0.25,
            save=True,
            generate_eda=False,
            model_dir=Path(cls.tmp.name),
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @property
    def bundle_path(self) -> Path:
        assert self.result.bundle_path is not None
        return self.result.bundle_path


class EncodeForBundleTests(PredictionTestBase):
    def test_matrix_matches_the_training_feature_count(self) -> None:
        bundle = load_bundle(self.bundle_path)
        matrix = encode_for_bundle(self.frame, bundle)

        self.assertEqual(matrix.shape, (len(self.frame), len(bundle["feature_names"])))
        self.assertFalse(bool(np.isnan(matrix).any()))

    def test_raw_headers_and_unseen_categories_are_handled(self) -> None:
        bundle = load_bundle(self.bundle_path)
        unseen = self.frame.head(3).copy()
        unseen["Category"] = "Brand New Category"
        unseen["Price"] = 999999.0

        matrix = encode_for_bundle(unseen, bundle)
        self.assertEqual(matrix.shape[0], 3)

    def test_missing_feature_list_raises(self) -> None:
        with self.assertRaises(PredictionError):
            encode_for_bundle(self.frame, {"model": None})


class PredictWithBundleTests(PredictionTestBase):
    def test_predictions_are_appended_and_labelled(self) -> None:
        scored = predict_with_bundle(self.frame, bundle=self.bundle_path)

        self.assertIn("prediction", scored.columns)
        self.assertIn("predicted_label", scored.columns)
        self.assertIn("confidence", scored.columns)
        self.assertEqual(len(scored), len(self.frame))

        original_labels = set(self.frame["Status"])
        self.assertTrue(set(scored["predicted_label"]).issubset(original_labels))
        self.assertTrue(((scored["confidence"] >= 0.0) & (scored["confidence"] <= 1.0)).all())

    def test_round_trip_accuracy_is_reasonable(self) -> None:
        scored = predict_with_bundle(self.frame, bundle=self.bundle_path)
        metrics = evaluate_predictions(self.frame["Status"], scored["predicted_label"])
        self.assertGreaterEqual(metrics["accuracy"], 0.6)

    def test_probability_columns_are_named_after_the_classes(self) -> None:
        scored = predict_with_bundle(self.frame, bundle=self.bundle_path)
        probability_columns = [c for c in scored.columns if c.startswith("probability_")]
        self.assertEqual(len(probability_columns), 3)
        self.assertAlmostEqual(float(scored[probability_columns].sum(axis=1).max()), 1.0, places=5)

    def test_probabilities_can_be_switched_off(self) -> None:
        scored = predict_with_bundle(
            self.frame, bundle=self.bundle_path, include_probabilities=False
        )
        self.assertNotIn("confidence", scored.columns)

    def test_bundle_without_a_model_raises(self) -> None:
        with self.assertRaises(PredictionError):
            predict_with_bundle(self.frame, bundle={"feature_names": ["price"], "model": None})


class PredictDataframeTests(PredictionTestBase):
    def test_scores_a_raw_frame_with_explicit_artefacts(self) -> None:
        prepared = prepare_dataset(self.frame, "status", test_size=0.25, random_state=7)
        result = run_training_pipeline(
            self.frame,
            "status",
            selected_models=["Decision Tree"],
            test_size=0.25,
            save=False,
            generate_eda=False,
            verbose=False,
        )
        predictions = predict_dataframe(
            result.best_model,
            self.frame,
            feature_names=prepared.feature_names,
            scaler=prepared.scaler,
            encoders=prepared.encoders,
        )
        self.assertEqual(len(predictions), len(self.frame))

    def test_scores_without_scaler_or_encoders(self) -> None:
        # Graceful degradation: with only the feature list the categorical
        # columns collapse to 0 but prediction still returns one row per record.
        result = run_training_pipeline(
            self.frame,
            "status",
            selected_models=["Decision Tree"],
            test_size=0.25,
            save=False,
            generate_eda=False,
            verbose=False,
        )
        predictions = predict_dataframe(
            result.best_model, self.frame, feature_names=result.prepared.feature_names
        )
        self.assertEqual(len(predictions), len(self.frame))


class MetadataAndComparisonTests(PredictionTestBase):
    def test_model_metadata_summary(self) -> None:
        metadata = model_metadata(self.bundle_path)
        self.assertEqual(metadata["target"], "status")
        self.assertEqual(metadata["n_features"], 7)
        self.assertIn("accuracy", metadata["metrics"])
        self.assertEqual(len(metadata["classes"]), 3)

    def test_model_metadata_from_a_loaded_bundle(self) -> None:
        metadata = model_metadata(load_bundle(self.bundle_path))
        self.assertEqual(metadata["model_name"], self.result.best_model_name)

    def test_evaluate_predictions_perfect_scores(self) -> None:
        metrics = evaluate_predictions(["a", "b", "a"], ["a", "b", "a"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_compare_saved_models_reports_accuracy(self) -> None:
        table = compare_saved_models(
            {"current": self.bundle_path}, self.frame, target_column="Status"
        )
        self.assertIn("current", table.index)
        self.assertIn("accuracy", table.columns)
        self.assertGreaterEqual(float(table.loc["current", "accuracy"]), 0.6)

    def test_compare_saved_models_without_ground_truth(self) -> None:
        table = compare_saved_models(
            {"current": self.bundle_path}, self.frame.drop(columns=["Status"])
        )
        self.assertTrue(table.empty or "accuracy" not in table.columns)


class BundleRoundTripTests(PredictionTestBase):
    """
    Regression tests for the scaler round-trip.

    The winner of :func:`run_training_pipeline` is refit on the *whole* dataset
    before it is persisted.  That matrix has to live in the same (standardised)
    space as the train/test splits, otherwise a reloaded bundle predicts with a
    completely different scale and the accuracy collapses.
    """

    def test_bundle_encoding_reproduces_the_prepared_matrix(self) -> None:
        bundle = load_bundle(self.bundle_path)
        prepared = prepare_dataset(self.frame, "status", test_size=0.25)

        matrix = encode_for_bundle(self.frame, bundle)

        self.assertEqual(matrix.shape, prepared.x_full.shape)
        np.testing.assert_allclose(matrix, prepared.x_full, atol=1e-8)

    def test_full_matrix_is_standardised(self) -> None:
        prepared = prepare_dataset(self.frame, "status", test_size=0.25)
        self.assertLess(float(np.abs(prepared.x_full.mean(axis=0)).max()), 0.2)

    def test_reloaded_bundle_keeps_its_training_accuracy(self) -> None:
        bundle = load_bundle(self.bundle_path)
        prepared = prepare_dataset(self.frame, "status", test_size=0.25)

        predictions = np.ravel(bundle["model"].predict(encode_for_bundle(self.frame, bundle)))
        scores = evaluate_predictions(prepared.y_full, predictions)

        self.assertGreaterEqual(scores["accuracy"], self.result.best_metrics["accuracy"] - 0.05)
        self.assertGreaterEqual(scores["accuracy"], 0.85)

    def test_predict_with_bundle_returns_one_label_per_row(self) -> None:
        scored = predict_with_bundle(self.frame, bundle=self.bundle_path)
        self.assertEqual(len(scored), len(self.frame))
        self.assertEqual(scored["predicted_label"].notna().sum(), len(self.frame))
        self.assertLessEqual(scored["predicted_label"].nunique(), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
