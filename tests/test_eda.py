"""
Unit tests for ``src.eda`` - the automated exploratory data analysis report.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tests.helpers import make_classification_frame, make_survey_frame
from src.eda import (
    build_eda_summary,
    generate_eda_report,
    generate_fallback_report,
)
from src import config


class EdaSummaryTests(unittest.TestCase):
    def test_summary_matches_the_report_cards(self) -> None:
        frame = make_classification_frame(rows=50)
        frame.loc[frame.index[0], "Price"] = None
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # one duplicate row

        summary = build_eda_summary(frame)
        self.assertEqual(summary["observations"], 51)
        self.assertEqual(summary["variables"], 10)
        self.assertEqual(summary["missing_cells"], 2)
        self.assertGreaterEqual(summary["duplicate_rows"], 1)
        self.assertGreater(summary["numeric_variables"], 0)
        self.assertGreater(summary["categorical_variables"], 0)
        self.assertRegex(summary["memory_size"], r"(B|KB|MB)")
        self.assertIn("variable_types", summary)

    def test_summary_of_an_empty_frame(self) -> None:
        summary = build_eda_summary(pd.DataFrame({"a": []}))
        self.assertEqual(summary["observations"], 0)
        self.assertEqual(summary["missing_percentage"], 0.0)
        self.assertEqual(summary["average_record_size"], "0 B")


class FallbackReportTests(unittest.TestCase):
    def test_report_contains_every_section(self) -> None:
        html = generate_fallback_report(make_classification_frame(rows=30), title="Survey EDA")
        for marker in (
            "Survey EDA",
            "Variable types",
            "Missing values per column",
            "Descriptive statistics (numeric)",
            "Correlation matrix (numeric)",
            "Data preview",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, html)

    def test_report_without_numeric_columns(self) -> None:
        html = generate_fallback_report(pd.DataFrame({"letter": ["a", "b", "c"]}))
        self.assertIn("No numeric columns.", html)


class GenerateReportTests(unittest.TestCase):
    def test_report_is_written_to_the_requested_path(self) -> None:
        frame = make_classification_frame(rows=40)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "eda.html"
            # Force the dependency-free path so the test is fast and deterministic.
            with mock.patch("src.eda.profile_report_available", return_value=False):
                written = generate_eda_report(frame, output_path=target)

            self.assertEqual(written, target)
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 0)
            self.assertIn("<html", target.read_text(encoding="utf-8").lower())

    def test_default_path_uses_the_reports_directory(self) -> None:
        frame = make_survey_frame(rows=40)
        with tempfile.TemporaryDirectory() as tmp:
            previous = config.EDA_REPORT_PATH
            config.EDA_REPORT_PATH = Path(tmp) / "eda_report.html"
            try:
                with mock.patch("src.eda.profile_report_available", return_value=False):
                    written = generate_eda_report(frame)
            finally:
                config.EDA_REPORT_PATH = previous
            self.assertEqual(written.name, "eda_report.html")
            self.assertTrue(written.exists())

    def test_large_frames_are_sampled(self) -> None:
        frame = make_classification_frame(rows=80)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sampled.html"
            with mock.patch("src.eda.profile_report_available", return_value=False):
                generate_eda_report(frame, output_path=target, sample_limit=20)

            html = target.read_text(encoding="utf-8")
            self.assertIn("random sample of 20 rows", html)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
