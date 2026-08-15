"""Spotify song analysis: a small, tested EDA pipeline.

Public API mirrors the notebook flow: load -> clean -> add features ->
analyse (scoring, hypothesis tests, co-occurrence).
"""

from __future__ import annotations

from .data import AUDIO_FEATURES, clean, load_raw
from .features import add_duration_bucket, add_features
from .analysis import (
    bayesian_weighted_score,
    compare_explicit,
    genre_cooccurrence,
    popularity_predictability,
)

__version__ = "0.1.0"

__all__ = [
    "AUDIO_FEATURES",
    "load_raw",
    "clean",
    "add_features",
    "add_duration_bucket",
    "bayesian_weighted_score",
    "compare_explicit",
    "genre_cooccurrence",
    "popularity_predictability",
]
