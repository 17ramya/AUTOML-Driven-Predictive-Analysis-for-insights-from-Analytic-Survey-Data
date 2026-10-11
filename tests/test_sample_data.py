"""
Unit tests for ``scripts.make_sample_data`` - the bundled dataset generator.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts import make_sample_data as sampler
from src import config


class GenerateSalesTests(unittest.TestCase):
    def test_columns_match_table_4_2(self) -> None:
        frame = sampler._generate_sales(rows=120, seed=1)
        self.assertEqual(
            list(frame.columns),
            [
                "Order ID",
                "Product",
                "Category",
                "Price",
                "Quantity",
                "Total Sales",
                "Order Date",
                "Customer Location",
                "Payment Method",
                "Status",
            ],
        )
        self.assertEqual(len(frame), 120)

    def test_status_has_three_learnable_classes(self) -> None:
        frame = sampler._generate_sales(rows=250, seed=42)
        self.assertEqual(set(frame["Status"]), {"Completed", "Pending", "Cancelled"})
        self.assertTrue(frame["Status"].value_counts().min() >= 10)

    def test_generation_is_deterministic(self) -> None:
        first = sampler._generate_sales(rows=60, seed=7)
        second = sampler._generate_sales(rows=60, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_no_missing_values_are_produced(self) -> None:
        frame = sampler._generate_sales(rows=60, seed=3)
        self.assertEqual(int(frame.isna().sum().sum()), 0)

    def test_total_sales_tracks_price_times_quantity(self) -> None:
        frame = sampler._generate_sales(rows=200, seed=5)
        self.assertGreater(frame["Total Sales"].corr(frame["Price"] * frame["Quantity"]), 0.95)


class GenerateListingsTests(unittest.TestCase):
    def test_columns_and_text_features(self) -> None:
        frame = sampler._generate_listings(rows=200, seed=2)
        for column in (
            "Posting Id",
            "Image",
            "Title",
            "Label Group",
            "title_prep",
            "pos",
            "word_count",
            "char_count",
            "avg_word_length",
        ):
            with self.subTest(column=column):
                self.assertIn(column, frame.columns)

        self.assertTrue((frame["word_count"] > 0).all())
        self.assertTrue((frame["char_count"] > 0).all())
        self.assertTrue(frame["avg_word_length"].between(1, 15).all())

    def test_label_group_has_several_classes(self) -> None:
        frame = sampler._generate_listings(rows=400, seed=9)
        self.assertGreaterEqual(frame["Label Group"].nunique(), 3)


class GenerateImagesTests(unittest.TestCase):
    def test_images_are_written_for_each_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            listings = sampler._generate_listings(rows=40, seed=4)
            written = sampler._generate_images(Path(tmp), listings, count=5, size=32, seed=4)
            files = sorted(Path(tmp).glob("*.png"))

        self.assertEqual(written, 5)
        self.assertEqual(len(files), 5)

    def test_missing_pillow_degrades_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            listings = sampler._generate_listings(rows=10, seed=4)
            with mock.patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
                written = sampler._generate_images(Path(tmp), listings, count=2, size=16, seed=1)
        self.assertEqual(written, 0)


class MainEntryPointTests(unittest.TestCase):
    def test_main_writes_all_three_artefacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.multiple(
                config,
                SAMPLE_DATA_DIR=root,
                SAMPLE_IMAGES_DIR=root / "images",
                SAMPLE_TABULAR_PATH=root / "sales.csv",
                SAMPLE_LISTINGS_PATH=root / "listings.csv",
            ):
                exit_code = sampler.main(
                    ["--rows", "80", "--listings-rows", "60", "--images", "3", "--seed", "11"]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "sales.csv").exists())
            self.assertTrue((root / "listings.csv").exists())
            self.assertEqual(len(list((root / "images").glob("*.png"))), 3)
            self.assertEqual(len(pd.read_csv(root / "sales.csv")), 80)

    def test_full_flag_uses_the_thesis_sizes(self) -> None:
        captured: dict[str, int] = {}
        original_sales = sampler._generate_sales
        original_listings = sampler._generate_listings

        def fake_sales(rows: int, seed: int) -> pd.DataFrame:
            captured["rows"] = rows
            return original_sales(rows=20, seed=seed)

        def fake_listings(rows: int, seed: int) -> pd.DataFrame:
            captured["listings_rows"] = rows
            return original_listings(rows=20, seed=seed)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.multiple(
                config,
                SAMPLE_DATA_DIR=root,
                SAMPLE_IMAGES_DIR=root / "images",
                SAMPLE_TABULAR_PATH=root / "sales.csv",
                SAMPLE_LISTINGS_PATH=root / "listings.csv",
            ), mock.patch.object(
                sampler, "_generate_sales", side_effect=fake_sales
            ), mock.patch.object(
                sampler, "_generate_listings", side_effect=fake_listings
            ), mock.patch.object(sampler, "_generate_images", return_value=0):
                sampler.main(["--full"])

        self.assertEqual(captured["rows"], 250)
        self.assertEqual(captured["listings_rows"], 32412)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
