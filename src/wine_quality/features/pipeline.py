"""The preprocessing contract shared by training and serving.

This module is the single source of truth for how raw measurements become model
input. Training wraps :func:`build_preprocessor` inside the estimator pipeline
that gets serialised, so the exact same fitted transformation is replayed at
inference time. That is what structurally prevents train/serve skew: there is no
second implementation for production to drift away from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wine_quality.config import (
    FEATURE_BOUNDS,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
)


class FeatureValidationError(ValueError):
    """Raised when a feature payload cannot be turned into a valid model input."""


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Build the preprocessing transformer.

    All predictors in this dataset are continuous, so the transformer is a
    single numeric branch. Median imputation is included as a defensive measure:
    the training set has no missing values, but a live request might, and it is
    better to fall back to the training median than to reject the request.

    Args:
        scale: Whether to standardise features. Enabled for linear and
            distance-based models, harmless for tree ensembles, and essential
            when comparing candidates on equal footing.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=steps)

    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, list(NUMERIC_FEATURES))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def to_frame(features: dict[str, float]) -> pd.DataFrame:
    """Convert a single feature mapping into a one-row dataframe.

    The API and dashboard both accept plain mappings. Column order is pinned to
    :data:`~wine_quality.config.FEATURE_NAMES` so that requests cannot introduce
    ordering bugs.

    Args:
        features: Mapping of feature name to measured value.

    Returns:
        A one-row dataframe with canonical column ordering.

    Raises:
        FeatureValidationError: If any required feature is missing.
    """
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise FeatureValidationError(f"Missing required features: {missing}")

    row = {name: features[name] for name in FEATURE_NAMES}
    return pd.DataFrame([row], columns=list(FEATURE_NAMES))


def validate_features(features: dict[str, float]) -> dict[str, float]:
    """Validate a feature mapping against the expected names and ranges.

    Args:
        features: Mapping of feature name to measured value.

    Returns:
        The mapping with values coerced to ``float``.

    Raises:
        FeatureValidationError: If features are missing, non-numeric, or outside
            the plausible physical bounds configured in
            :data:`~wine_quality.config.FEATURE_BOUNDS`.
    """
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise FeatureValidationError(f"Missing required features: {missing}")

    unexpected = [name for name in features if name not in FEATURE_NAMES]
    if unexpected:
        raise FeatureValidationError(f"Unexpected features: {unexpected}")

    coerced: dict[str, float] = {}
    for name in FEATURE_NAMES:
        try:
            value = float(features[name])
        except (TypeError, ValueError) as exc:
            raise FeatureValidationError(f"Feature '{name}' is not numeric") from exc

        if not np.isfinite(value):
            raise FeatureValidationError(f"Feature '{name}' must be a finite number")

        low, high = FEATURE_BOUNDS[name]
        if not low <= value <= high:
            raise FeatureValidationError(
                f"Feature '{name}' value {value} is outside the plausible range [{low}, {high}]"
            )

        coerced[name] = value

    return coerced


def feature_ranges() -> dict[str, dict[str, float | str]]:
    """Expose plausible input ranges for API schemas and dashboard widgets.

    Descriptions are included so a single call gives clients everything they
    need to render an input form.

    Returns:
        Mapping of feature name to its ``low``, ``high``, and ``description``.
    """
    return {
        name: {
            "low": low,
            "high": high,
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
        }
        for name, (low, high) in FEATURE_BOUNDS.items()
    }


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix over features and target.

    Used by the dashboard's EDA page to show which physicochemical properties
    move with perceived quality.

    Args:
        frame: Dataframe containing at least the feature columns.

    Returns:
        Correlation matrix restricted to columns present in the frame.
    """
    columns = [name for name in (*FEATURE_NAMES, "quality") if name in frame.columns]
    return frame[columns].corr(numeric_only=True) if columns else pd.DataFrame()