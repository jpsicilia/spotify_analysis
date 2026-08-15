"""Analytical routines: artist/genre scoring, hypothesis tests, co-occurrence."""

from __future__ import annotations

import logging
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def bayesian_weighted_score(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "popularity",
    prior_strength: float | None = None,
) -> pd.DataFrame:
    """Rank groups by a shrinkage-weighted mean (IMDB Top-250 style).

    This replaces the earlier ``mean_pop * log(n)`` weighting, which had a
    fatal flaw: ``log(1) = 0`` sent every single-song artist to a score of
    exactly zero, silently dropping ~59% of the artists from the ranking
    regardless of how popular their one song was.

    The shrinkage estimator instead pulls each group's mean toward the global
    mean in proportion to how little data supports it::

        score = (n * group_mean + m * global_mean) / (n + m)

    where ``m`` (``prior_strength``) is the number of "pseudo-observations" of
    the global mean. A group with many songs keeps its own mean; a group with
    one song is pulled toward the global average rather than deleted or
    trusted blindly. Every group gets a finite, comparable score.

    Args:
        df: DataFrame containing ``group_col`` and ``value_col``.
        group_col: Column to group by (e.g. ``"artist"`` or ``"genre_clean"``).
        value_col: Numeric column to average (default ``"popularity"``).
        prior_strength: ``m`` above. If ``None``, defaults to the median group
            size, a common data-driven choice.

    Returns:
        DataFrame with columns ``[group_col, n_songs, mean_value, score]``,
        sorted by ``score`` descending.
    """
    global_mean = df[value_col].mean()
    grouped = df.groupby(group_col)[value_col]
    n = grouped.size()
    mean = grouped.mean()

    m = float(n.median()) if prior_strength is None else float(prior_strength)

    score = (n * mean + m * global_mean) / (n + m)

    out = (
        pd.DataFrame(
            {group_col: n.index, "n_songs": n.values, "mean_value": mean.values,
             "score": score.values}
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "Scored %d groups of '%s' (prior m=%.1f, global mean=%.1f)",
        len(out), group_col, m, global_mean,
    )
    return out


def compare_explicit(
    df: pd.DataFrame, value_col: str = "popularity"
) -> dict[str, float]:
    """Test whether explicit tracks differ in popularity from non-explicit.

    Reports a Mann-Whitney U test (no normality assumption), the rank-biserial
    effect size, and a 95% bootstrap CI for the difference in means. This turns
    "the bars look different" into a defensible statistical statement.

    Args:
        df: DataFrame with a boolean/0-1 ``explicit`` column and ``value_col``.
        value_col: Column to compare (default ``"popularity"``).

    Returns:
        Dict with group means, U statistic, p-value, effect size, and the
        bootstrap CI bounds for (explicit_mean - non_explicit_mean).
    """
    explicit = df.loc[df["explicit"].astype(bool), value_col].to_numpy()
    non_explicit = df.loc[~df["explicit"].astype(bool), value_col].to_numpy()

    u_stat, p_value = stats.mannwhitneyu(explicit, non_explicit, alternative="two-sided")
    # Rank-biserial effect size from U.
    effect = 1 - (2 * u_stat) / (len(explicit) * len(non_explicit))

    rng = np.random.default_rng(42)
    diffs = np.empty(5000)
    for i in range(5000):
        a = rng.choice(explicit, size=len(explicit), replace=True)
        b = rng.choice(non_explicit, size=len(non_explicit), replace=True)
        diffs[i] = a.mean() - b.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    result = {
        "mean_explicit": float(explicit.mean()),
        "mean_non_explicit": float(non_explicit.mean()),
        "n_explicit": int(len(explicit)),
        "n_non_explicit": int(len(non_explicit)),
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "effect_size_rank_biserial": float(effect),
        "diff_ci95_low": float(ci_low),
        "diff_ci95_high": float(ci_high),
    }
    logger.info("Explicit vs non-explicit: p=%.4g, effect=%.3f", p_value, effect)
    return result


def popularity_predictability(
    df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5
) -> pd.DataFrame:
    """Can audio features predict popularity? Cross-validated R^2.

    Uses ``KFold(shuffle=True)`` deliberately: the cleaned DataFrame is sorted
    by popularity, so an unshuffled split would leak the target's structure
    across folds and produce absurd (large negative) R^2. Shuffling gives an
    honest estimate. A linear baseline is scaled; the tree model is not (trees
    are scale-invariant).

    Args:
        df: DataFrame with ``feature_cols`` and a ``popularity`` column.
        feature_cols: Audio-feature column names to use as predictors.
        n_splits: Number of CV folds.

    Returns:
        DataFrame with one row per model: ``[model, r2_mean, r2_std]``.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[feature_cols].to_numpy()
    y = df["popularity"].to_numpy()
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        rows.append({"model": name, "r2_mean": scores.mean(), "r2_std": scores.std()})
        logger.info("%s: R2 = %.3f +/- %.3f", name, scores.mean(), scores.std())
    return pd.DataFrame(rows)


def genre_cooccurrence(df: pd.DataFrame) -> Counter:
    """Count unordered co-occurring genre pairs across all tracks.

    Args:
        df: DataFrame with a ``genre_list`` column (list of genres per row).

    Returns:
        Counter mapping frozenset-like tuples ``(g1, g2)`` to counts.
    """
    pairs: list[tuple[str, str]] = []
    for genres in df["genre_list"]:
        unique = sorted(set(genres))
        if len(unique) > 1:
            pairs.extend(combinations(unique, 2))
    return Counter(pairs)
