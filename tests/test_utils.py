"""
Unit tests for ``src.utils`` - the shared, dependency-free helpers.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import make_classification_frame
from src.utils import (
    categorical_columns,
    dataframe_memory_usage,
    dataframe_to_html,
    ensure_directory,
    format_metrics,
    get_file_extension,
    human_readable_bytes,
    is_categorical,
    is_continuous_numeric,
    numeric_columns,
    read_text_file,
    round_metrics,
    safe_filename,
    standardize_column_name,
    standardize_column_names,
)


class StandardizeColumnNameTests(unittest.TestCase):
    def test_removes_symbols_spaces_and_casing(self) -> None:
        cases = {
            " Total Sales ($) ": "total_sales",
            "Order ID": "order_id",
            "Payment   Method": "payment_method",
            "Customer Location": "customer_location",
            "Cafe-Sales": "cafe_sales",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(standardize_column_name(raw), expected)

    def test_keeps_unicode_letters(self) -> None:
        # `\w` is Unicode aware, so accented letters survive the cleanup.
        self.assertEqual(standardize_column_name("Café-Sales"), "café_sales")

    def test_handles_non_string_input(self) -> None:
        self.assertEqual(standardize_column_name(None), "")
        self.assertEqual(standardize_column_name(7), "7")

    def test_dataframe_headers_become_unique_snake_case(self) -> None:
        frame = pd.DataFrame([[1, 2, 3, 4]], columns=["A", "a", "", "A "])
        cleaned = standardize_column_names(frame)
        self.assertEqual(list(cleaned.columns), ["a", "a_1", "column_2", "a_2"])

    def test_original_frame_is_not_mutated_by_default(self) -> None:
        frame = pd.DataFrame([[1]], columns=["Order ID"])
        standardize_column_names(frame)
        self.assertEqual(list(frame.columns), ["Order ID"])


class FileHelperTests(unittest.TestCase):
    def test_get_file_extension(self) -> None:
        self.assertEqual(get_file_extension("data/report.CSV"), "csv")
        self.assertEqual(get_file_extension(Path("book.xlsx")), "xlsx")
        self.assertEqual(get_file_extension("no_extension"), "")

    def test_safe_filename(self) -> None:
        self.assertEqual(safe_filename("Random Forest"), "Random_Forest")
        self.assertEqual(safe_filename("Extra Trees (tuned)"), "Extra_Trees_tuned")
        self.assertEqual(safe_filename("!!!###"), "")

    def test_ensure_directory_creates_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a" / "b" / "c"
            self.assertEqual(ensure_directory(target), target)
            self.assertTrue(target.is_dir())

    def test_read_text_file_missing_and_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.txt"
            self.assertEqual(read_text_file(missing), "")

            existing = Path(tmp) / "yes.txt"
            existing.write_text("hello", encoding="utf-8")
            self.assertEqual(read_text_file(existing), "hello")


class DataFrameHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_classification_frame(rows=40)

    def test_is_categorical(self) -> None:
        self.assertTrue(is_categorical(self.frame["Status"]))
        self.assertTrue(is_categorical(pd.Series([True, False])))
        self.assertFalse(is_categorical(self.frame["Price"]))

    def test_is_continuous_numeric(self) -> None:
        # Only a column with fractional values is "continuous" for scikit-learn;
        # integer codes and non-numeric columns are valid classification targets.
        self.assertTrue(is_continuous_numeric(self.frame["Price"]))
        self.assertTrue(is_continuous_numeric(pd.Series([0.5, 1.0, 2.0])))

        self.assertFalse(is_continuous_numeric(self.frame["Status"]))
        self.assertFalse(is_continuous_numeric(self.frame["Quantity"]))
        self.assertFalse(is_continuous_numeric(pd.Series([0, 1, 2])))
        self.assertFalse(is_continuous_numeric(pd.Series([0.0, 1.0, 2.0])))
        self.assertFalse(is_continuous_numeric(pd.Series([True, False])))
        self.assertFalse(is_continuous_numeric(pd.Series([float("nan"), float("nan")])))

    def test_numeric_and_categorical_columns(self) -> None:
        numeric = numeric_columns(self.frame)
        self.assertIn("Price", numeric)
        self.assertNotIn("Status", numeric)

        categorical = categorical_columns(self.frame, exclude=["Order Date"])
        self.assertIn("Status", categorical)
        self.assertNotIn("Order Date", categorical)

    def test_memory_usage_and_html_preview(self) -> None:
        self.assertRegex(dataframe_memory_usage(self.frame), r"\d+\.\d+ (B|KB|MB|GB)")
        html = dataframe_to_html(self.frame, title="Preview", max_rows=3)
        self.assertIn("Preview", html)
        self.assertIn("Shape: 40 rows x 10 columns", html)

    def test_human_readable_bytes(self) -> None:
        self.assertEqual(human_readable_bytes(512), "512.0 B")
        self.assertEqual(human_readable_bytes(2048), "2.0 KB")


class MetricHelperTests(unittest.TestCase):
    def test_round_metrics_keeps_non_numeric_values(self) -> None:
        rounded = round_metrics({"accuracy": 0.123456, "model": "Extra Trees", "flag": True})
        self.assertEqual(rounded["accuracy"], 0.1235)
        self.assertEqual(rounded["model"], "Extra Trees")
        self.assertIs(rounded["flag"], True)

    def test_format_metrics(self) -> None:
        text = format_metrics({"accuracy": 0.5, "model": "KNN"}, digits=2)
        self.assertEqual(text, "accuracy: 0.50 | model: KNN")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
