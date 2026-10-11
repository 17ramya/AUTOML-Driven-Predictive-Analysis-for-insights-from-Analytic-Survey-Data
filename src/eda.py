"""
MODULE 1 (part c) - Automated Exploratory Data Analysis (EDA)
=============================================================
Thesis reference: Section 4.3.2 (code listing) and Figure 5.2 / 5.3.

The module wraps ``ydata-profiling`` (the successor of ``pandas_profiling``
used in the report).  When the library is unavailable - or the dataset is too
small to profile - a lightweight, dependency-free HTML report is produced
instead so that the pipeline never breaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .preprocessing import detect_column_types
from .utils import dataframe_memory_usage, ensure_directory, human_readable_bytes

__all__ = [
    "build_eda_summary",
    "generate_eda_report",
    "generate_fallback_report",
    "profile_report_available",
]


# --------------------------------------------------------------------------- #
# Dataset summary ("Dataset statistics" panel of Figure 5.2)
# --------------------------------------------------------------------------- #
def build_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Structured summary of the dataset used by the dashboard header cards.

    Keys mirror the wording of Figure 5.2: number of variables, observations,
    missing cells, duplicate rows, memory size and the variable-type breakdown.
    """
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    variable_types = detect_column_types(df)
    memory_bytes = int(df.memory_usage(deep=True).sum())

    return {
        "observations": int(df.shape[0]),
        "variables": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_percentage": round((missing_cells / total_cells) * 100, 2) if total_cells else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_size": dataframe_memory_usage(df),
        "average_record_size": human_readable_bytes(memory_bytes / df.shape[0]) if df.shape[0] else "0 B",
        "numeric_variables": len(variable_types["numeric"]),
        "categorical_variables": len(variable_types["categorical"]),
        "datetime_variables": len(variable_types["datetime"]),
        "variable_types": variable_types,
    }


def profile_report_available() -> bool:
    """True when ``ydata_profiling`` (or legacy ``pandas_profiling``) is importable."""
    try:  # pragma: no cover - depends on the environment
        import ydata_profiling  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        try:
            import pandas_profiling  # noqa: F401

            return True
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# Fallback report
# --------------------------------------------------------------------------- #
def generate_fallback_report(df: pd.DataFrame, title: str = "EDA Report") -> str:
    """
    Build a self-contained HTML report using only pandas.

    Contains the dataset summary, per-column statistics, missing-value counts
    and the correlation matrix of the numeric variables.
    """
    summary = build_eda_summary(df)
    numeric = df.select_dtypes(include="number")

    parts: list[str] = [
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2933}",
        "h1,h2{color:#0b3d91} table{border-collapse:collapse;margin-bottom:20px}",
        "th,td{border:1px solid #cbd5e1;padding:6px 10px;font-size:13px}",
        "th{background:#eef2f7} .card{display:inline-block;border:1px solid #cbd5e1;",
        "border-radius:8px;padding:10px 16px;margin:6px;background:#f8fafc}",
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]

    cards = {
        "Observations": summary["observations"],
        "Variables": summary["variables"],
        "Missing cells": summary["missing_cells"],
        "Duplicate rows": summary["duplicate_rows"],
        "Memory size": summary["memory_size"],
        "Average record size": summary["average_record_size"],
    }
    parts.append("<div>")
    for label, value in cards.items():
        parts.append(f"<div class='card'><b>{label}</b><br>{value}</div>")
    parts.append("</div>")

    breakdown = (
        f"numeric: {summary['numeric_variables']}, "
        f"categorical: {summary['categorical_variables']}, "
        f"datetime: {summary['datetime_variables']}"
    )
    parts.append(f"<h2>Variable types</h2><p>{breakdown}</p>")

    parts.append("<h2>Missing values per column</h2>")
    missing = df.isna().sum().rename("missing").to_frame()
    missing["missing_percent"] = (missing["missing"] / max(len(df), 1) * 100).round(2)
    parts.append(missing.to_html())

    parts.append("<h2>Descriptive statistics (numeric)</h2>")
    parts.append(numeric.describe().T.round(4).to_html() if not numeric.empty else "<p>No numeric columns.</p>")

    parts.append("<h2>Descriptive statistics (categorical)</h2>")
    categorical = df.select_dtypes(exclude="number")
    if categorical.empty:
        parts.append("<p>No categorical columns.</p>")
    else:
        describe = categorical.describe().T
        parts.append(describe.to_html())

    parts.append("<h2>Correlation matrix (numeric)</h2>")
    if numeric.shape[1] >= 2:
        parts.append(numeric.corr(numeric_only=True).round(3).to_html())
    else:
        parts.append("<p>Need at least two numeric columns for a correlation matrix.</p>")

    parts.append("<h2>Data preview</h2>")
    parts.append(df.head(20).to_html())

    parts.append("</body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def generate_eda_report(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "EDA Report",
    minimal: bool = True,
    explorative: bool = True,
    sample_limit: int = 20000,
) -> Path:
    """
    Generate the automated EDA report and write it to ``output_path``.

    Parameters
    ----------
    df:
        Dataset to profile.
    output_path:
        Destination ``.html`` file (defaults to ``reports/eda_report.html``).
    minimal:
        ``True`` keeps the report small - identical to the thesis code listing
        (``ProfileReport(df, title="EDA Report", explorative=True, minimal=True)``).
    sample_limit:
        Very large frames are profiled on a random sample to keep the runtime
        reasonable; the sample size is reported inside the HTML.

    Returns
    -------
    pathlib.Path
        Path of the written HTML report.
    """
    target = Path(output_path) if output_path is not None else config.EDA_REPORT_PATH
    ensure_directory(target.parent)

    frame = df
    note = ""
    if len(frame) > sample_limit:
        frame = frame.sample(n=sample_limit, random_state=config.RANDOM_STATE)
        note = f"<p><b>Note:</b> profiling performed on a random sample of {sample_limit:,} rows.</p>"

    if profile_report_available():
        try:
            from ydata_profiling import ProfileReport  # type: ignore

            profile = ProfileReport(
                frame, title=title, explorative=explorative, minimal=minimal
            )
            profile.to_file(str(target))
            if note:
                html = target.read_text(encoding="utf-8").replace("</body>", f"{note}</body>")
                target.write_text(html, encoding="utf-8")
            return target
        except Exception:  # pragma: no cover - fall back to the lightweight report
            pass

    html = generate_fallback_report(frame, title=title)
    if note:
        html = html.replace("</body>", f"{note}</body>")
    target.write_text(html, encoding="utf-8")
    return target

