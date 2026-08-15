"""Data loading and cleaning for the Spotify songs dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that carry the audio-feature signal, for convenience elsewhere.
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw songs CSV.

    Args:
        csv_path: Path to ``songs.csv``.

    Returns:
        Raw DataFrame, untouched.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], csv_path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset.

    Steps, in order:
      1. Drop exact duplicate rows.
      2. Collapse (artist, song) duplicates, keeping the most popular version.
      3. Replace the literal ``"set()"`` genre placeholder with ``"desconocido"``.

    The function is pure: it returns a new DataFrame and does not mutate the input.

    Args:
        df: Raw DataFrame from :func:`load_raw`.

    Returns:
        Cleaned DataFrame with a fresh RangeIndex.
    """
    df = df.copy()

    n_exact = df.duplicated().sum()
    df = df.drop_duplicates()

    n_pair = df.duplicated(subset=["artist", "song"]).sum()
    df = (
        df.sort_values("popularity", ascending=False)
        .drop_duplicates(subset=["artist", "song"], keep="first")
    )

    # A handful of rows carry the literal string "set()" instead of a genre.
    df["genre"] = df["genre"].replace("set()", "desconocido")

    df = df.reset_index(drop=True)
    logger.info(
        "Cleaning removed %d exact dups and %d artist+song dups -> %d rows",
        n_exact,
        n_pair,
        len(df),
    )
    return df
