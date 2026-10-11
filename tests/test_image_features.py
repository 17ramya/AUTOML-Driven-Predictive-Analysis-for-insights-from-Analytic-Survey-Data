"""
Unit tests for ``src.image_features`` - the multi-modal visual descriptors.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.image_features import (
    IMAGE_FEATURE_COLUMNS,
    ImageFeatureError,
    build_multimodal_dataset,
    extract_features_from_directory,
    extract_image_features,
    image_pillow_available,
    merge_image_features,
)


def _write_image(path: Path, size: int = 32, colour: tuple[int, int, int] = (10, 120, 200)) -> Path:
    from PIL import Image

    Image.new("RGB", (size, size), colour).save(path)
    return path


class ArrayFeatureTests(unittest.TestCase):
    def test_descriptor_columns_and_ranges(self) -> None:
        array = np.zeros((20, 40, 3), dtype=np.uint8)
        array[..., 0] = 255
        features = extract_image_features(array)

        self.assertEqual(set(features), set(IMAGE_FEATURE_COLUMNS))
        self.assertEqual(features["width"], 40.0)
        self.assertEqual(features["height"], 20.0)
        self.assertAlmostEqual(features["aspect_ratio"], 2.0, places=3)
        self.assertAlmostEqual(features["mean_r"], 1.0, places=3)
        self.assertAlmostEqual(features["mean_g"], 0.0, places=3)
        self.assertEqual(features["n_channels"], 3.0)

    def test_grayscale_arrays_are_supported(self) -> None:
        features = extract_image_features(np.full((16, 16), 128, dtype=np.uint8))
        self.assertEqual(features["n_channels"], 1.0)
        self.assertAlmostEqual(features["brightness"], 128 / 255, places=4)

    def test_degenerate_images_raise(self) -> None:
        with self.assertRaises(ImageFeatureError):
            extract_image_features(np.zeros((0, 5), dtype=np.uint8))


class FileFeatureTests(unittest.TestCase):
    def test_extract_from_a_png_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_image(Path(tmp) / "sample.png", size=24, colour=(255, 0, 0))
            features = extract_image_features(path)

        self.assertEqual(features["width"], 24.0)
        self.assertAlmostEqual(features["mean_r"], 1.0, places=3)
        self.assertGreater(features["file_size_kb"], 0.0)

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ImageFeatureError):
                extract_image_features(Path(tmp) / "nope.png")


class DirectoryFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        for index in range(3):
            _write_image(
                self.directory / f"listing_{index}.png",
                colour=(index * 60, 100, 200 - index * 20),
            )
        (self.directory / "notes.txt").write_text("ignored", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_one_row_per_image_and_only_images(self) -> None:
        frame = extract_features_from_directory(self.directory)
        self.assertEqual(len(frame), 3)
        self.assertIn("image", frame.columns)
        self.assertTrue(frame["error"].isna().all())
        for column in IMAGE_FEATURE_COLUMNS:
            self.assertIn(column, frame.columns)

    def test_limit_is_respected(self) -> None:
        frame = extract_features_from_directory(self.directory, limit=2)
        self.assertEqual(len(frame), 2)

    def test_empty_directory_returns_empty_frame_with_schema(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            frame = extract_features_from_directory(other)
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns)[0], "image")

    def test_unknown_directory_raises(self) -> None:
        with self.assertRaises(ImageFeatureError):
            extract_features_from_directory(self.directory / "missing")

    def test_pillow_is_available_in_this_environment(self) -> None:
        self.assertTrue(image_pillow_available())


class MergeTests(DirectoryFeatureTests):
    def setUp(self) -> None:
        super().setUp()
        self.listings = pd.DataFrame(
            {
                "posting_id": [f"PL{index}" for index in range(3)],
                "image": [f"listing_{index}.png" for index in range(3)],
                "title": ["widget", "gadget", "gizmo"],
                "label_group": [0, 1, 0],
            }
        )

    def test_merge_adds_the_visual_features(self) -> None:
        features = extract_features_from_directory(self.directory, filename_column="image")
        merged = merge_image_features(self.listings, features)

        self.assertEqual(len(merged), 3)
        self.assertIn("brightness", merged.columns)
        self.assertFalse(merged["brightness"].isna().any())

    def test_merge_fills_missing_images(self) -> None:
        listings = pd.concat(
            [self.listings, pd.DataFrame({"image": ["ghost.png"], "title": ["x"]})],
            ignore_index=True,
        )
        features = extract_features_from_directory(self.directory, filename_column="image")
        merged = merge_image_features(listings, features)
        self.assertEqual(len(merged), 4)
        self.assertFalse(merged["contrast"].isna().any())

    def test_merge_requires_the_join_keys(self) -> None:
        with self.assertRaises(ImageFeatureError):
            merge_image_features(self.listings, pd.DataFrame({"other": [1]}))
        with self.assertRaises(ImageFeatureError):
            merge_image_features(pd.DataFrame({"other": [1]}), pd.DataFrame({"image": ["a"]}))

    def test_build_multimodal_dataset_drops_the_error_column(self) -> None:
        merged = build_multimodal_dataset(self.listings, self.directory)
        self.assertNotIn("error", merged.columns)
        self.assertIn("brightness", merged.columns)
        self.assertEqual(len(merged), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
