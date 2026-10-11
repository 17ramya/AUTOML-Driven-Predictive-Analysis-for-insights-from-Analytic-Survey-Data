"""
Unit tests for ``src.preprocessing`` - type detection, imputation, encoding,
scaling and the end-to-end ``prepare_dataset`` orchestration.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.helpers import make_classification_frame, make_survey_frame
from src.preprocessing import (
    PreprocessingError,
    detect_column_types,
    detect_id_columns,
    encode_categorical_features,
    handle_missing_values,
    prepare_dataset,
    resolve_column,
    scale_features,
    split_features_target,
)
from src.utils import standardize_column_names


class ResolveColumnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_classification_frame(rows=20)

    def test_resolves_snake_case_against_raw_headers(self) -> None:
        self.assertEqual(resolve_column(self.frame, "total_sales"), "Total Sales ($)")
        self.assertEqual(resolve_column(self.frame, "order_id"), "Order ID")
        self.assertEqual(resolve_column(self.frame, "CUSTOMER LOCATION"), "Customer Location")

    def test_exact_match_wins(self) -> None:
        self.assertEqual(resolve_column(self.frame, "Status"), "Status")

    def test_unknown_column_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            resolve_column(self.frame, "not_a_column")


class DetectTypesTests(unittest.TestCase):
    def test_detects_numeric_categorical_and_datetime(self) -> None:
        frame = pd.DataFrame(
            {
                "price": [1.0, 2.0, 3.0],
                "category": ["a", "b", "a"],
                "order_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                "flag": [True, False, True],
            }
        )
        detected = detect_column_types(frame)
        self.assertEqual(detected["numeric"], ["price"])
        self.assertCountEqual(detected["categorical"], ["category", "flag"])
        self.assertEqual(detected["datetime"], ["order_date"])


class DetectIdColumnsTests(unittest.TestCase):
    def test_flags_identifiers_and_unique_text(self) -> None:
        frame = make_classification_frame(rows=60)
        identifiers = detect_id_columns(frame)

        self.assertIn("Order ID", identifiers)
        self.assertNotIn("Price", identifiers)
        self.assertNotIn("Status", identifiers)

    def test_numeric_columns_are_never_treated_as_identifiers(self) -> None:
        frame = pd.DataFrame({"value": np.arange(100, dtype=float), "target": [0, 1] * 50})
        self.assertEqual(detect_id_columns(frame), [])


class MissingValueTests(unittest.TestCase):
    def test_mean_imputation_removes_all_gaps(self) -> None:
        frame = make_survey_frame(rows=60)
        cleaned = handle_missing_values(frame, numeric_strategy="mean")
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)

    def test_median_strategy_is_supported(self) -> None:
        frame = make_survey_frame(rows=60)
        cleaned = handle_missing_values(frame, numeric_strategy="median")
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)

    def test_entirely_empty_column_gets_placeholder(self) -> None:
        frame = pd.DataFrame({"note": [None, None, None], "value": [1.0, 2.0, 3.0]})
        cleaned = handle_missing_values(frame)
        self.assertTrue((cleaned["note"] == "Unknown").all())


class EncodingTests(unittest.TestCase):
    def test_label_encoding_returns_fitted_encoders(self) -> None:
        frame = standardize_column_names(
            make_classification_frame(rows=40).drop(columns=["Order ID", "Order Date"])
        )
        encoded, encoders = encode_categorical_features(frame, method="label")

        self.assertEqual(encoded.select_dtypes(include="number").shape[1], encoded.shape[1])
        self.assertIn("category", encoders)
        self.assertEqual(encoders["category"]["type"], "label")
        self.assertTrue(hasattr(encoders["category"]["encoder"], "classes_"))

    def test_dict_encoding_returns_a_mapping(self) -> None:
        frame = pd.DataFrame({"colour": ["red", "green", "red"]})
        encoded, encoders = encode_categorical_features(frame, method="onehot")
        self.assertEqual(sorted(encoders["colour"]["mapping"]), ["green", "red"])
        self.assertTrue(encoded["colour"].notna().all())

    def test_invalid_method_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            encode_categorical_features(pd.DataFrame({"a": ["x"]}), method="nonsense")

    def test_excluded_columns_are_left_alone(self) -> None:
        frame = pd.DataFrame({"colour": ["red", "green"], "keep": ["x", "y"]})
        encoded, encoders = encode_categorical_features(frame, method="label", exclude=["keep"])
        self.assertEqual(list(encoders), ["colour"])
        self.assertEqual(encoded["keep"].tolist(), ["x", "y"])


class ScalingTests(unittest.TestCase):
    def test_training_split_is_standardised(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
        scaled, _, scaler = scale_features(frame)
        self.assertAlmostEqual(float(np.abs(scaled.mean(axis=0)).max()), 0.0, places=6)
        self.assertAlmostEqual(float(scaled.std(axis=0).max()), 1.0, places=6)
        self.assertTrue(hasattr(scaler, "mean_"))

    def test_reusing_a_fitted_scaler_does_not_refit(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        test = pd.DataFrame({"a": [4.0, 5.0]})
        _, _, scaler = scale_features(train)
        _, scaled_test, reused = scale_features(train, test, scaler=scaler)
        self.assertIs(scaler, reused)
        self.assertEqual(scaled_test.shape, (2, 1))

    def test_missing_test_split_returns_none(self) -> None:
        scaled, test_scaled, _ = scale_features(pd.DataFrame({"a": [1.0, 2.0]}))
        self.assertIsNone(test_scaled)
        self.assertEqual(scaled.shape, (2, 1))


class SplitTests(unittest.TestCase):
    def test_drops_target_and_requested_columns(self) -> None:
        frame = make_classification_frame(rows=30)
        x, y = split_features_target(frame, "Status", drop_columns=["Order ID"])
        self.assertNotIn("Status", x.columns)
        self.assertNotIn("Order ID", x.columns)
        self.assertEqual(len(y), len(frame))

    def test_missing_drop_columns_are_ignored(self) -> None:
        frame = make_classification_frame(rows=30)
        x, _ = split_features_target(frame, "Status", drop_columns=["not_there"])
        self.assertEqual(x.shape[1], frame.shape[1] - 1)

    def test_error_when_no_features_remain(self) -> None:
        frame = pd.DataFrame({"Status": ["a", "b"]})
        with self.assertRaises(PreprocessingError):
            split_features_target(frame, "Status")


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_classification_frame(rows=200)
        self.prepared = prepare_dataset(self.frame, "status", test_size=0.25, random_state=42)

    def test_identifier_columns_are_excluded_from_the_features(self) -> None:
        self.assertNotIn("order_id", self.prepared.feature_names)
        self.assertNotIn("order_date", self.prepared.feature_names)
        self.assertNotIn("status", self.prepared.feature_names)
        self.assertEqual(self.prepared.n_features, 7)

    def test_split_sizes_and_scaling(self) -> None:
        self.assertEqual(self.prepared.x_train.shape[0], 150)
        self.assertEqual(self.prepared.x_test.shape[0], 50)
        self.assertEqual(self.prepared.x_train.shape[1], self.prepared.n_features)
        self.assertFalse(bool(np.isnan(self.prepared.x_train).any()))
        self.assertLess(float(np.abs(self.prepared.x_train.mean(axis=0)).max()), 1e-6)

    def test_target_is_label_encoded_and_reversible(self) -> None:
        self.assertEqual(self.prepared.n_classes, 3)
        self.assertIsNotNone(self.prepared.label_encoder)
        restored = self.prepared.label_encoder.inverse_transform(
            self.prepared.y_full.to_numpy().astype(int)
        )
        self.assertCountEqual(list(restored), self.frame["Status"].astype(str).tolist())
        self.assertEqual(self.prepared.class_labels, ["Cancelled", "Completed", "Pending"])

    def test_summary_dictionary(self) -> None:
        summary = self.prepared.summary()
        self.assertEqual(summary["target"], "status")
        self.assertEqual(summary["classes"], 3)
        self.assertEqual(summary["train_rows"], 150)
        self.assertEqual(summary["test_rows"], 50)

    def test_auto_drop_ids_can_be_disabled(self) -> None:
        prepared = prepare_dataset(
            self.frame,
            "status",
            drop_columns=(),
            test_size=0.25,
            random_state=42,
            auto_drop_ids=False,
        )
        self.assertIn("order_id", prepared.feature_names)
        self.assertIn("order_date", prepared.feature_names)
        self.assertEqual(prepared.n_features, 9)

    def test_explicit_drop_columns_always_win(self) -> None:
        prepared = prepare_dataset(
            self.frame,
            "status",
            drop_columns=("product",),
            test_size=0.25,
            random_state=42,
        )
        self.assertNotIn("product", prepared.feature_names)
        self.assertEqual(prepared.n_features, 6)

    def test_unknown_target_raises(self) -> None:
        with self.assertRaises(PreprocessingError):
            prepare_dataset(self.frame, "does_not_exist")

    def test_numeric_target_is_left_numeric(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
        prepared = prepare_dataset(frame, "b", test_size=0.25, random_state=0)
        self.assertIsNone(prepared.label_encoder)
        self.assertEqual(prepared.n_classes, 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
