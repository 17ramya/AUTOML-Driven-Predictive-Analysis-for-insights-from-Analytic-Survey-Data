"""
MULTI-MODAL SUPPORT - Image feature extraction
=============================================
Thesis reference: Abstract and Section 4.2.2 - "the system supports structured
tabular data and unstructured image datasets, enabling multi-modal learning
with over 30,000 image-based records".

Design note
-----------
Deep CNN embeddings would require a GPU and a large download, so this module
extracts a compact, deterministic **visual descriptor** per image (geometry,
colour statistics and texture energy).  Those descriptors can be joined with
the tabular listing metadata and fed to the AutoML pipeline just like any other
numeric feature - which is exactly how the `product listings` dataset of the
report is modelled (text + image -> label group).

Pillow (PIL) is used for decoding; if it is unavailable the functions raise a
clear :class:`ImageFeatureError` instead of crashing the whole application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .utils import ensure_directory

__all__ = [
    "ImageFeatureError",
    "IMAGE_FEATURE_COLUMNS",
    "extract_image_features",
    "extract_features_from_directory",
    "merge_image_features",
    "image_pillow_available",
]

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

IMAGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "width",
    "height",
    "aspect_ratio",
    "n_channels",
    "mean_r",
    "mean_g",
    "mean_b",
    "brightness",
    "contrast",
    "edge_density",
    "colourfulness",
    "file_size_kb",
)


class ImageFeatureError(RuntimeError):
    """Raised when Pillow is missing or an image cannot be decoded."""


def image_pillow_available() -> bool:
    """True when Pillow can be imported."""
    try:  # pragma: no cover - environment dependent
        from PIL import Image  # noqa: F401

        return True
    except Exception:
        return False


def _features_from_array(array: np.ndarray, file_size_kb: float = 0.0) -> dict[str, float]:
    """Compute the visual descriptor of an ``H x W`` or ``H x W x C`` array."""
    if array.ndim == 2:
        gray = array.astype(float)
        rgb = np.dstack([gray, gray, gray])
    else:
        rgb = array[..., :3].astype(float)
        gray = rgb.mean(axis=2)

    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        raise ImageFeatureError("Image has a zero dimension.")

    if rgb.max() > 1.5:  # typical uint8 range
        rgb = rgb / 255.0
        gray = gray / 255.0

    mean_r, mean_g, mean_b = (float(rgb[..., i].mean()) for i in range(3))
    brightness = float(gray.mean())
    contrast = float(gray.std())

    horizontal = np.abs(np.diff(gray, axis=1)).mean() if width > 1 else 0.0
    vertical = np.abs(np.diff(gray, axis=0)).mean() if height > 1 else 0.0
    edge_density = float((horizontal + vertical) / 2.0)

    # Hasler & Süsstrunk style colourfulness proxy.
    rg = np.abs(rgb[..., 0] - rgb[..., 1])
    yb = np.abs(0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2])
    colourfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": round(width / height, 6),
        "n_channels": float(1 if array.ndim == 2 else array.shape[2]),
        "mean_r": round(mean_r, 6),
        "mean_g": round(mean_g, 6),
        "mean_b": round(mean_b, 6),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "edge_density": round(edge_density, 6),
        "colourfulness": round(colourfulness, 6),
        "file_size_kb": round(float(file_size_kb), 3),
    }


def extract_image_features(source: str | Path | "np.ndarray") -> dict[str, float]:
    """
    Extract the visual descriptor of a single image.

    ``source`` may be a path to an image file or an ``H x W [x C]`` numpy array
    (already decoded), which keeps the function usable from notebooks/tests.
    """
    if isinstance(source, np.ndarray):
        return _features_from_array(source)

    path = Path(source)
    if not path.exists():
        raise ImageFeatureError(f"Image not found: {path}")

    try:
        from PIL import Image  # noqa: WPS433 (deliberate local import)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImageFeatureError(
            "Pillow is required for image feature extraction. Install it with `pip install Pillow`."
        ) from exc

    try:
        with Image.open(path) as handle:
            array = np.asarray(handle.convert("RGB"))
    except Exception as exc:
        raise ImageFeatureError(f"Could not decode '{path}': {exc}") from exc

    return _features_from_array(array, file_size_kb=path.stat().st_size / 1024.0)


def extract_features_from_directory(
    directory: str | Path,
    limit: int | None = None,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
    filename_column: str = "image",
) -> pd.DataFrame:
    """
    Walk ``directory`` and return one row of features per image.

    Parameters
    ----------
    limit:
        Optional cap on the number of images processed - useful for the very
        large listings dataset mentioned in the report (> 30 000 records).
    filename_column:
        Name of the column holding the file name (``"image"`` in the report).

    Returns
    -------
    pandas.DataFrame
        Columns: ``filename_column`` + :data:`IMAGE_FEATURE_COLUMNS` + ``error``.
    """
    base = Path(directory)
    if not base.exists():
        raise ImageFeatureError(f"Directory not found: {base}")

    allowed = tuple(ext.lower() for ext in extensions)
    files: list[Path] = sorted(
        path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )
    if limit is not None:
        files = files[: int(limit)]

    rows: list[dict[str, object]] = []
    for path in files:
        record: dict[str, object] = {filename_column: path.name}
        try:
            record.update(extract_image_features(path))
            record["error"] = None
        except ImageFeatureError as exc:
            record["error"] = str(exc)
        rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[filename_column, *IMAGE_FEATURE_COLUMNS, "error"])
    return frame


def merge_image_features(
    tabular_df: pd.DataFrame,
    image_df: pd.DataFrame,
    left_on: str = "image",
    right_on: str = "image",
    how: str = "left",
) -> pd.DataFrame:
    """
    Join the visual descriptors onto the tabular listing metadata.

    This is the "multi-modal" step: the resulting frame contains the text
    features *and* the image features and can be fed straight into
    :func:`src.model_training.run_training_pipeline`.
    """
    if right_on not in image_df.columns:
        raise ImageFeatureError(
            f"Image feature frame must contain a '{right_on}' column. "
            f"Available: {', '.join(map(str, image_df.columns))}"
        )
    if left_on not in tabular_df.columns:
        raise ImageFeatureError(f"Tabular frame has no '{left_on}' column to join on.")

    features = [c for c in IMAGE_FEATURE_COLUMNS if c in image_df.columns]
    right = image_df[[right_on, *features]].drop_duplicates(subset=[right_on])
    merged = tabular_df.merge(right, left_on=left_on, right_on=right_on, how=how)

    for column in features:
        if column in merged.columns:
            merged[column] = merged[column].fillna(merged[column].median(numeric_only=True))

    return merged


def build_multimodal_dataset(
    listings_df: pd.DataFrame,
    image_directory: str | Path,
    limit: int | None = None,
    left_on: str = "image",
) -> pd.DataFrame:
    """
    Convenience helper: extract descriptors for ``image_directory`` and merge
    them onto ``listings_df``.
    """
    features = extract_features_from_directory(image_directory, limit=limit, filename_column=left_on)
    merged = merge_image_features(listings_df, features, left_on=left_on, right_on=left_on)
    return merged.drop(columns=["error"], errors="ignore")
