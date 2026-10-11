"""
Sample data generator
=====================
Creates the two datasets described in the report so the project can be run
end-to-end without any external download:

* ``data/sample/amazon_sales_data_2025.csv``
  Table 4.2 - structured sales transactions
  (Order ID, Product, Category, Price, Quantity, Total Sales, Order Date,
   Customer Location, Payment Method, Status).

* ``data/sample/product_listings.csv``  (+ ``data/sample/images/*.png``)
  Table 4.3/4.4 - product listings with text metadata and an image reference,
  the multi-modal dataset used for the image experiments.

Usage
-----
    python scripts/make_sample_data.py                    # demo sized (default)
    python scripts/make_sample_data.py --full             # thesis sized
                                                          # (250 sales / 32 412 listings)
    python scripts/make_sample_data.py --rows 500 --listings-rows 5000 --images 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.utils import ensure_directory  # noqa: E402

CATEGORIES = ["Electronics", "Home & Kitchen", "Books", "Fashion", "Sports", "Beauty"]
CITIES = ["Houston", "Miami", "Dallas", "Austin", "Seattle", "Boston", "Chicago", "Denver", "Phoenix", "Atlanta"]
PAYMENTS = ["Credit Card", "Debit Card", "PayPal", "UPI", "Net Banking"]

PRODUCTS = {
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "Power Bank", "LED Monitor"],
    "Home & Kitchen": ["Air Fryer", "Coffee Maker", "Blender", "Cookware Set", "Vacuum Cleaner"],
    "Books": ["Data Science Handbook", "AutoML in Practice", "Python Cookbook", "ML Yearbook", "Deep Learning"],
    "Fashion": ["Running Shoes", "Denim Jacket", "Leather Wallet", "Sunglasses", "Backpack"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Cricket Bat", "Football", "Cycling Helmet"],
    "Beauty": ["Face Serum", "Hair Dryer", "Perfume", "Sunscreen SPF50", "Lip Balm Pack"],
}


def _generate_sales(rows: int, seed: int) -> pd.DataFrame:
    """Build the Amazon-style sales survey dataset of Table 4.2."""
    rng = np.random.default_rng(seed)

    category = rng.choice(CATEGORIES, size=rows)
    product = [str(rng.choice(PRODUCTS[c])) for c in category]
    price = np.round(rng.uniform(15, 1200, size=rows), 2)
    quantity = rng.integers(1, 9, size=rows)
    location = rng.choice(CITIES, size=rows)
    payment = rng.choice(PAYMENTS, size=rows)

    # Total Sales is a near-deterministic function of price x quantity, which is
    # why the report observes a strong Price/Total-Sales correlation (Fig 5.4/5.5).
    noise = rng.normal(0, 25, size=rows)
    total_sales = np.round(np.clip(price * quantity + noise, 10, None), 2)

    # Status is a learnable function of the order profile plus a little label noise.
    demand = (quantity - quantity.min()) / max(float(quantity.max() - quantity.min()), 1.0)
    value = price / price.max()
    affinity = np.array([CATEGORIES.index(c) for c in category]) / (len(CATEGORIES) - 1)
    risk = 0.5 * value + 0.3 * affinity - 0.4 * demand + rng.normal(0, 0.015, size=rows)

    status = np.where(risk > 0.34, "Cancelled", np.where(risk > 0.16, "Pending", "Completed"))

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 180, size=rows), unit="D")

    return pd.DataFrame(
        {
            "Order ID": [f"ORD{100000 + i}" for i in range(rows)],
            "Product": product,
            "Category": category,
            "Price": price,
            "Quantity": quantity,
            "Total Sales": total_sales,
            "Order Date": dates.strftime("%Y-%m-%d"),
            "Customer Location": location,
            "Payment Method": payment,
            "Status": status,
        }
    )


def _generate_listings(rows: int, seed: int) -> pd.DataFrame:
    """Build the multi-modal product-listing metadata table (Table 4.4)."""
    rng = np.random.default_rng(seed + 7)

    category = rng.choice(CATEGORIES, size=rows)
    base_titles = [str(rng.choice(PRODUCTS[c])) for c in category]
    adjectives = ["Original", "Premium", "Pack of 2", "Brand New", "Imported", "Certified"]
    titles = [f"{base} {rng.choice(adjectives)} {rng.integers(1, 999)}" for base in base_titles]

    frame = pd.DataFrame(
        {
            "Posting Id": [f"PL{500000 + i}" for i in range(rows)],
            "Image": [f"listing_{i:05d}.png" for i in range(rows)],
            "Title": titles,
        }
    )

    frame["title_prep"] = frame["Title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    frame["pos"] = [
        " ".join("NN" if len(word) > 3 else "JJ" for word in title.split())
        for title in frame["title_prep"]
    ]
    frame["word_count"] = frame["title_prep"].str.split().str.len()
    frame["char_count"] = frame["title_prep"].str.len()
    frame["avg_word_length"] = np.round(frame["char_count"] / frame["word_count"].replace(0, 1), 4)

    # `Label Group` (the duplicate-listing target) is derived from the text
    # statistics and the category, so the classifier has a genuine signal.
    category_index = np.array([CATEGORIES.index(c) for c in category])
    signature = (
        frame["word_count"].to_numpy() * 3
        + frame["char_count"].to_numpy() // 10
        + category_index * 5
    )
    groups = max(3, min(12, rows // 40))
    frame["Label Group"] = (signature % groups).astype(int)

    return frame[
        [
            "Posting Id",
            "Image",
            "Title",
            "Label Group",
            "title_prep",
            "pos",
            "word_count",
            "char_count",
            "avg_word_length",
        ]
    ]


def _generate_images(directory: Path, listings: pd.DataFrame, count: int, size: int, seed: int) -> int:
    """Render simple synthetic product thumbnails (one per listing)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover - Pillow missing
        print("Pillow is not installed - skipping image generation.")
        return 0

    ensure_directory(directory)
    rng = np.random.default_rng(seed + 11)
    written = 0

    for filename in listings["Image"].head(count):
        palette = rng.integers(40, 230, size=3)
        canvas = Image.new("RGB", (size, size), tuple(int(v) for v in palette))
        draw = ImageDraw.Draw(canvas)
        for _ in range(int(rng.integers(3, 8))):
            x0, y0 = (int(v) for v in rng.integers(0, max(size - 10, 11), size=2))
            w, h = (int(v) for v in rng.integers(8, max(size // 2, 9), size=2))
            colour = tuple(int(v) for v in rng.integers(0, 255, size=3))
            box = [x0, y0, x0 + w, y0 + h]
            if int(rng.integers(0, 2)) == 0:
                draw.rectangle(box, fill=colour)
            else:
                draw.ellipse(box, fill=colour)
        canvas.save(directory / str(filename))
        written += 1

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the sample datasets for the AutoML project.")
    parser.add_argument("--rows", type=int, default=250, help="rows of the sales dataset (thesis: 250)")
    parser.add_argument("--listings-rows", type=int, default=1200, help="rows of the listings dataset")
    parser.add_argument("--images", type=int, default=24, help="number of synthetic images to render")
    parser.add_argument("--image-size", type=int, default=96, help="image width/height in pixels")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE, help="random seed")
    parser.add_argument("--full", action="store_true", help="generate thesis-sized datasets (250 / 32 412)")
    args = parser.parse_args(argv)

    if args.full:
        args.rows = 250
        args.listings_rows = 32412
        args.images = 60

    ensure_directory(config.SAMPLE_DATA_DIR)
    ensure_directory(config.SAMPLE_IMAGES_DIR)

    sales = _generate_sales(args.rows, args.seed)
    sales.to_csv(config.SAMPLE_TABULAR_PATH, index=False)
    print(f"[sales]    {config.SAMPLE_TABULAR_PATH}  ({sales.shape[0]} rows x {sales.shape[1]} cols)")

    listings = _generate_listings(args.listings_rows, args.seed)
    listings.to_csv(config.SAMPLE_LISTINGS_PATH, index=False)
    print(f"[listings] {config.SAMPLE_LISTINGS_PATH}  ({listings.shape[0]} rows x {listings.shape[1]} cols)")

    written = _generate_images(config.SAMPLE_IMAGES_DIR, listings, args.images, args.image_size, args.seed)
    print(f"[images]   {config.SAMPLE_IMAGES_DIR}  ({written} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
