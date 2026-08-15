"""Tests for the spotify_analysis pipeline.

Run from the repo root:  pytest tests/ -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spotify_analysis as sa

DATA = Path(__file__).resolve().parents[1] / "data" / "songs.csv"


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    df = sa.clean(sa.load_raw(DATA))
    return sa.add_features(df)


def test_clean_removes_duplicates(clean_df):
    """No (artist, song) duplicates should remain after cleaning."""
    assert clean_df.duplicated(subset=["artist", "song"]).sum() == 0


def test_clean_is_pure():
    """clean() must not mutate its input."""
    raw = sa.load_raw(DATA)
    before = len(raw)
    _ = sa.clean(raw)
    assert len(raw) == before


def test_genre_clean_is_sorted():
    """genre_clean must collapse permutations (rock,pop == pop,rock)."""
    df = pd.DataFrame({
        "genre": ["rock, pop", "pop, rock"],
        "duration_ms": [200000, 200000],
    })
    out = sa.add_features(df)
    assert out["genre_clean"].nunique() == 1
    assert out["genre_clean"].iloc[0] == "pop, rock"


def test_bayesian_score_includes_single_song_groups(clean_df):
    """The regression that log(n) introduced: single-song artists were dropped.

    Every artist must receive a finite, non-zero score now.
    """
    scores = sa.bayesian_weighted_score(clean_df, "artist")
    single = scores[scores["n_songs"] == 1]
    assert len(single) > 0                    # they exist
    assert (single["score"] > 0).all()        # and are not zeroed out
    assert np.isfinite(scores["score"]).all()


def test_bayesian_score_shrinks_toward_global(clean_df):
    """A single-song group should sit between its own value and the global mean."""
    scores = sa.bayesian_weighted_score(clean_df, "artist")
    single = scores[scores["n_songs"] == 1].iloc[0]
    global_mean = clean_df["popularity"].mean()
    lo, hi = sorted([single["mean_value"], global_mean])
    assert lo <= single["score"] <= hi


def test_compare_explicit_keys(clean_df):
    res = sa.compare_explicit(clean_df)
    for key in ("p_value", "effect_size_rank_biserial", "diff_ci95_low", "diff_ci95_high"):
        assert key in res
    assert 0.0 <= res["p_value"] <= 1.0


def test_cooccurrence_pairs_are_unique_unordered(clean_df):
    counts = sa.genre_cooccurrence(clean_df)
    for (g1, g2) in counts:
        assert g1 < g2      # canonical ordering, no (b,a) duplicates
