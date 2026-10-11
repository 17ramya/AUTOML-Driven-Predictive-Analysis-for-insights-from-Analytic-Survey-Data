"""
Unit tests for ``src.data_ingestion`` (load, validate, persist).
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import make_classification_frame
from src.data_ingestion import (
    DataIngestionError,
    ingest_dataset,
    is_supported_file,
    load_dataset,
    preview_dataset,
    save_source_data,
)


class FileValidationTests(unittest.TestCase):
    def test_supported_extensions(self) -> None:
        for name in ("data.csv", "DATA.CSV", "book.xlsx", "legacy.xls"):
            with self.subTest(name=name):
                self.assertTrue(is_supported_file(name))

    def test_unsupported_extensions(self) -> None:
        for name in ("data.txt", "data.json", "archive.zip", "no_extension"):
            with self.subTest(name=name):
                self.assertFalse(is_supported_file(name))


class LoadDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_classification_frame(rows=30)

    def test_load_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "survey.csv"
            self.frame.to_csv(path, index=False)

            loaded = load_dataset(path)
            self.assertEqual(loaded.shape, self.frame.shape)
            self.assertEqual(list(loaded.columns), list(self.frame.columns))

    def test_load_from_uploaded_file_object(self) -> None:
        buffer = io.BytesIO(self.frame.to_csv(index=False).encode("utf-8"))
        buffer.name = "survey.csv"  # streamlit uploads carry a name
        loaded = load_dataset(buffer)
        self.assertEqual(loaded.shape, self.frame.shape)

    def test_load_from_buffer_with_explicit_filename(self) -> None:
        buffer = io.BytesIO(self.frame.to_csv(index=False).encode("utf-8"))
        loaded = load_dataset(buffer, filename="survey.csv")
        self.assertEqual(loaded["Status"].nunique(), self.frame["Status"].nunique())

    def test_missing_filename_raises(self) -> None:
        with self.assertRaises(DataIngestionError):
            load_dataset(io.BytesIO(b"a,b\n1,2\n"))

    def test_unsupported_format_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("hello", encoding="utf-8")
            with self.assertRaises(DataIngestionError):
                load_dataset(path)


class IngestDatasetTests(unittest.TestCase):
    def test_ingest_standardises_headers_and_saves_file(self) -> None:
        frame = make_classification_frame(rows=40)
        buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "source_data.csv"
            cleaned, saved_path = ingest_dataset(buffer, filename="survey.csv", save_path=destination)

            self.assertEqual(saved_path, destination)
            self.assertTrue(destination.exists())
            self.assertIn("total_sales", cleaned.columns)
            self.assertIn("order_id", cleaned.columns)
            self.assertNotIn("Total Sales ($)", cleaned.columns)

    def test_ingest_keeps_numeric_columns_numeric(self) -> None:
        frame = make_classification_frame(rows=40)
        frame.loc[frame.index[:4], "Price"] = None
        buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            cleaned, _ = ingest_dataset(
                buffer, filename="survey.csv", save_path=Path(tmp) / "source.csv"
            )

        # Numeric gaps must stay numeric for the imputation step of the pipeline.
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["price"]))
        self.assertTrue(cleaned["price"].isna().any())

    def test_ingest_fills_categorical_gaps_with_placeholder(self) -> None:
        frame = make_classification_frame(rows=40)
        frame.loc[frame.index[:3], "Category"] = None
        buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            cleaned, _ = ingest_dataset(
                buffer, filename="survey.csv", save_path=Path(tmp) / "source.csv"
            )

        self.assertFalse(cleaned["category"].isna().any())
        self.assertIn("Unknown", set(cleaned["category"]))

    def test_empty_dataset_raises(self) -> None:
        buffer = io.BytesIO(b"order_id,status\n")
        with self.assertRaises(DataIngestionError):
            ingest_dataset(buffer, filename="empty.csv")

    def test_unsupported_extension_raises(self) -> None:
        buffer = io.BytesIO(b"a,b\n1,2\n")
        with self.assertRaises(DataIngestionError):
            ingest_dataset(buffer, filename="data.txt")


class PersistenceAndPreviewTests(unittest.TestCase):
    def test_save_source_data_round_trip(self) -> None:
        frame = make_classification_frame(rows=20)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_source_data(frame, Path(tmp) / "out.csv")
            self.assertTrue(path.exists())
            self.assertEqual(pd.read_csv(path).shape, frame.shape)

    def test_preview_dataset_summary(self) -> None:
        frame = make_classification_frame(rows=20)
        preview = preview_dataset(frame, rows=3)

        self.assertEqual(preview["shape"], (20, 10))
        self.assertEqual(len(preview["head"]), 3)
        self.assertIn("Status", preview["columns"])
        self.assertIn("Price", preview["dtypes"])
        self.assertGreater(preview["memory_usage"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
