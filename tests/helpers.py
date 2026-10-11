"""
Shared fixtures for the unit-test suite.

Every fixture is generated with a fixed seed so the suite is deterministic and
fast (no network access, no file downloads).  Import them with::

    from tests.helpers import make_classification_frame, make_survey_frame

from the repository root, e.g.::

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``import src...`` work no matter where the tests are launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "amazon_sales_data_2025.csv"

CATEGORIES = (
    "Books",
    "Beauty",
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Sports",
)
PRODUCTS = (
    "Wireless Mouse",
    "Yoga Mat",
    "Novel",
    "Face Serum",
    "Bluetooth Speaker",
    "Running Shoes",
    "Cookware Set",
    "Cricket Bat",
)
LOCATIONS = ("Chennai", "Mumbai", "Delhi", "Bengaluru", "Hyderabad")
PAYMENTS = ("Credit Card", "UPI", "Net Banking", "Cash on Delivery")
STATUS_LABELS = ("Completed", "Pending", "Cancelled")


def make_classification_frame(rows: int = 300, seed: int = 7) -> pd.DataFrame:
    """
    A small, *learnable* 3-class classification frame (headers deliberately
    kept in the human-readable form of the thesis Table 4.2 so that the
    header-standardisation logic is exercised).
    """
    rng = np.random.default_rng(seed)

    price = np.round(rng.uniform(20.0, 500.0, rows), 2)
    quantity = rng.integers(1, 6, rows)
    category = rng.choice(CATEGORIES, size=rows)

    risk = (
        0.6 * (price / price.max())
        - 0.4 * (quantity - quantity.min()) / (quantity.max() - quantity.min())
        + rng.normal(0.0, 0.03, rows)
    )
    status = np.where(risk > 0.36, "Cancelled", np.where(risk > 0.20, "Pending", "Completed"))

    return pd.DataFrame(
        {
            "Order ID": [f"B{100000 + index}" for index in range(rows)],
            "Order Date": pd.date_range("2025-01-01", periods=rows, freq="h").astype(str),
            "Status": status,
            "Category": category,
            "Product": rng.choice(PRODUCTS, size=rows),
            "Quantity": quantity,
            "Price": price,
            "Total Sales ($)": np.round(price * quantity, 2),
            "Customer Location": rng.choice(LOCATIONS, size=rows),
            "Payment Method": rng.choice(PAYMENTS, size=rows),
        }
    )


def make_survey_frame(rows: int = 120, seed: int = 11) -> pd.DataFrame:
    """Minimal frame with missing values, used for the imputation tests."""
    rng = np.random.default_rng(seed)
    frame = make_classification_frame(rows=rows, seed=seed)
    frame.loc[frame.index[:5], "Price"] = np.nan
    frame.loc[frame.index[5:9], "Category"] = None
    return frame


def make_binary_frame(rows: int = 160, seed: int = 3) -> pd.DataFrame:
    """Two-class, purely numeric frame (no identifiers, no missing values)."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, rows)
    noise = rng.normal(0.0, 1.0, (rows, 2))
    return pd.DataFrame(
        {
            "feature_one": signal,
            "feature_two": noise[:, 0],
            "feature_three": noise[:, 1],
            "outcome": (signal > 0.0).astype(int),
        }
    )


def load_sample_sales_frame() -> pd.DataFrame:
    """Load the generated sample dataset, skipping the test when absent."""
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLE_DATA_PATH} not found. Run `python scripts/make_sample_data.py` first."
        )
    return pd.read_csv(SAMPLE_DATA_PATH)
