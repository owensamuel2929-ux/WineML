"""Conversion of the raw sensory score into supervised learning targets.

The dataset is framed two ways, as the original authors note:

1. **Regression** — predict the ordinal sensory score directly (3-8 observed).
2. **Binary classification** — apply an arbitrary cutoff so that wines scoring
   at or above it are labelled "good". The Kaggle data card suggests 7, which
   labels roughly 13.6% of rows positive and is the framing used here.
"""

from __future__ import annotations

import pandas as pd

from wine_quality.config import FEATURE_NAMES, TARGET_NAME, Settings, get_settings


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the continuous regression target.

    Args:
        frame: Validated dataset.

    Returns:
        Tuple of the feature frame (canonical column order) and the quality series.
    """
    return frame[list(FEATURE_NAMES)].copy(), frame[TARGET_NAME].copy()


def make_binary_target(
    quality: pd.Series,
    cutoff: int | None = None,
    settings: Settings | None = None,
) -> pd.Series:
    """Binarise the quality score into a "good wine" indicator.

    Args:
        quality: Raw quality scores.
        cutoff: Score at or above which a wine is "good".
        settings: Settings supplying the default cutoff.

    Returns:
        Integer series named ``is_good`` holding 1 for good wines and 0 otherwise.
    """
    settings = settings or get_settings()
    threshold = cutoff if cutoff is not None else settings.good_quality_cutoff
    return (quality >= threshold).astype(int).rename("is_good")


def build_targets(
    frame: pd.DataFrame,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Produce both targets for a single dataset in one pass.

    Keeps the feature frame shared between tasks so the two models are trained
    on provably identical inputs, which makes their metrics comparable.

    Args:
        frame: Validated dataset.
        settings: Settings supplying the cutoff.

    Returns:
        Tuple of ``(features, quality, is_good)``.
    """
    settings = settings or get_settings()
    features, quality = split_features_target(frame)
    return features, quality, make_binary_target(quality, settings=settings)