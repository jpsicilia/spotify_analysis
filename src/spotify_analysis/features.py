"""Feature engineering: derived columns used across the analysis."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_genre_string(raw: str) -> str:
    """Normalize one comma-separated genre string.

    Splits on commas, strips whitespace, lowercases, sorts alphabetically,
    and rejoins. Sorting makes ``"pop, rock"`` and ``"rock, pop"`` collapse to
    a single canonical label, so they group together downstream.
    """
    parts = [g.strip().lower() for g in str(raw).split(",")]
    return ", ".join(sorted(parts))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns.

    Adds:
      - ``duration_min``: track length in minutes (single source of truth).
      - ``genre_clean``: canonical, sorted, lowercased genre label.
      - ``genre_list``: list of individual genres (for co-occurrence analysis).

    Args:
        df: Cleaned DataFrame from :func:`spotify_analysis.data.clean`.

    Returns:
        New DataFrame with the extra columns.
    """
    df = df.copy()

    df["duration_min"] = df["duration_ms"] / 60_000

    df["genre_clean"] = df["genre"].apply(_normalize_genre_string)

    # Assign directly on the aligned index (no dropna round-trip, which could
    # silently misalign rows). str(x) guards against any non-string entries.
    df["genre_list"] = df["genre"].apply(
        lambda x: [g.strip().lower() for g in str(x).split(",")]
    )

    logger.info("Added duration_min, genre_clean, genre_list")
    return df


def add_duration_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket ``duration_min`` into interpretable ranges.

    Uses ``duration_min`` (created in :func:`add_features`) as the single
    source of truth — no second, redundant duration column.
    """
    df = df.copy()
    bins = [0, 2.5, 3.5, 4.5, df["duration_min"].max()]
    labels = ["<2.5 min", "2.5-3.5 min", "3.5-4.5 min", ">4.5 min"]
    df["duration_bucket"] = pd.cut(
        df["duration_min"], bins=bins, labels=labels, include_lowest=True
    )
    return df
