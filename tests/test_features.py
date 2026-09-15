"""Tests for the shared preprocessing contract.

These are the most important tests in the suite: the preprocessor is serialised
inside the model pipeline, so any change here silently changes what production
computes. They pin the behaviour that training and serving both depend on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wine_quality.config import FEATURE_BOUNDS, FEATURE_NAMES
from wine_quality.features.pipeline import (
    FeatureValidationError,
    build_preprocessor,
    correlation_matrix,
    feature_ranges,
    to_frame,
    validate_features,
)


class TestBuildPreprocessor:
    """The transformer must accept the canonical schema and scale correctly."""

    def test_fits_and_transforms_canonical_frame(self, raw_frame: pd.DataFrame) -> None:
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(raw_frame[list(FEATURE_NAMES)])
        assert transformed.shape == (len(raw_frame), len(FEATURE_NAMES))

    def test_scaled_output_is_standardised(self, raw_frame: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(scale=True)
        transformed = preprocessor.fit_transform(raw_frame[list(FEATURE_NAMES)])
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-8)

    def test_unscaled_output_preserves_original_scale(self, raw_frame: pd.DataFrame) -> None:
        preprocessor = build_preprocessor(scale=False)
        transformed = preprocessor.fit_transform(raw_frame[list(FEATURE_NAMES)])
        assert np.allclose(transformed.mean(axis=0), raw_frame[list(FEATURE_NAMES)].mean(), atol=1e-6)

    def test_imputes_missing_values_with_training_median(self, raw_frame: pd.DataFrame) -> None:
        """A live request may omit a value; the training median is the fallback."""
        preprocessor = build_preprocessor(scale=False)
        preprocessor.fit(raw_frame[list(FEATURE_NAMES)])

        incomplete = raw_frame[list(FEATURE_NAMES)].iloc[[0]].copy()
        incomplete.loc[:, "alcohol"] = np.nan
        transformed = preprocessor.transform(incomplete)

        assert not np.isnan(transformed).any()
        expected = raw_frame["alcohol"].median()
        assert np.isclose(transformed[0, list(FEATURE_NAMES).index("alcohol")], expected)

    def test_transform_is_deterministic(self, raw_frame: pd.DataFrame) -> None:
        """Same input must always produce the same output — no hidden randomness."""
        preprocessor = build_preprocessor()
        preprocessor.fit(raw_frame[list(FEATURE_NAMES)])
        first = preprocessor.transform(raw_frame[list(FEATURE_NAMES)].head(10))
        second = preprocessor.transform(raw_frame[list(FEATURE_NAMES)].head(10))
        np.testing.assert_array_equal(first, second)

    def test_ignores_extra_columns(self, raw_frame: pd.DataFrame) -> None:
        """The target column must not leak into the feature matrix."""
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(raw_frame)
        assert transformed.shape[1] == len(FEATURE_NAMES)


class TestToFrame:
    """Single-sample conversion must pin column order."""

    def test_produces_one_row_with_canonical_order(self, sample_features: dict[str, float]) -> None:
        frame = to_frame(sample_features)
        assert frame.shape == (1, len(FEATURE_NAMES))
        assert list(frame.columns) == list(FEATURE_NAMES)

    def test_column_order_independent_of_dict_order(self, sample_features: dict[str, float]) -> None:
        reversed_features = dict(reversed(list(sample_features.items())))
        frame = to_frame(reversed_features)
        assert list(frame.columns) == list(FEATURE_NAMES)
        assert frame.iloc[0]["alcohol"] == sample_features["alcohol"]

    def test_missing_feature_raises(self, sample_features: dict[str, float]) -> None:
        incomplete = {key: value for key, value in sample_features.items() if key != "alcohol"}
        with pytest.raises(FeatureValidationError, match="Missing required features"):
            to_frame(incomplete)


class TestValidateFeatures:
    """Validation is the API's first line of defence against bad input."""

    def test_accepts_valid_payload(self, sample_features: dict[str, float]) -> None:
        assert validate_features(sample_features) == sample_features

    def test_coerces_integers_to_float(self, sample_features: dict[str, float]) -> None:
        payload = {**sample_features, "alcohol": 9}
        assert isinstance(validate_features(payload)["alcohol"], float)

    def test_rejects_missing_feature(self, sample_features: dict[str, float]) -> None:
        incomplete = {key: value for key, value in sample_features.items() if key != "pH"}
        with pytest.raises(FeatureValidationError, match="Missing required features"):
            validate_features(incomplete)

    def test_rejects_unexpected_feature(self, sample_features: dict[str, float]) -> None:
        with pytest.raises(FeatureValidationError, match="Unexpected features"):
            validate_features({**sample_features, "grape variety": 1.0})

    def test_rejects_non_numeric_value(self, sample_features: dict[str, float]) -> None:
        with pytest.raises(FeatureValidationError, match="not numeric"):
            validate_features({**sample_features, "alcohol": "strong"})

    def test_rejects_nan(self, sample_features: dict[str, float]) -> None:
        with pytest.raises(FeatureValidationError, match="finite"):
            validate_features({**sample_features, "alcohol": float("nan")})

    def test_rejects_infinity(self, sample_features: dict[str, float]) -> None:
        with pytest.raises(FeatureValidationError, match="finite"):
            validate_features({**sample_features, "alcohol": float("inf")})

    @pytest.mark.parametrize("feature", list(FEATURE_NAMES))
    def test_rejects_values_below_lower_bound(
        self, sample_features: dict[str, float], feature: str
    ) -> None:
        low, _ = FEATURE_BOUNDS[feature]
        with pytest.raises(FeatureValidationError, match="outside the plausible range"):
            validate_features({**sample_features, feature: low - 1.0})

    @pytest.mark.parametrize("feature", list(FEATURE_NAMES))
    def test_rejects_values_above_upper_bound(
        self, sample_features: dict[str, float], feature: str
    ) -> None:
        _, high = FEATURE_BOUNDS[feature]
        with pytest.raises(FeatureValidationError, match="outside the plausible range"):
            validate_features({**sample_features, feature: high + 1.0})

    def test_accepts_boundary_values(self, sample_features: dict[str, float]) -> None:
        """Bounds are inclusive — a value exactly at the limit is valid."""
        payload = {**sample_features, "pH": FEATURE_BOUNDS["pH"][0]}
        assert validate_features(payload)["pH"] == FEATURE_BOUNDS["pH"][0]


class TestFeatureRanges:
    """The API publishes these ranges so clients can build input forms."""

    def test_covers_every_feature(self) -> None:
        assert set(feature_ranges().keys()) == set(FEATURE_NAMES)

    def test_low_is_below_high(self) -> None:
        for name, bounds in feature_ranges().items():
            assert bounds["low"] < bounds["high"], f"{name} has an inverted range"


class TestCorrelationMatrix:
    """The dashboard's EDA page depends on this shape."""

    def test_includes_target_and_features(self, raw_frame: pd.DataFrame) -> None:
        matrix = correlation_matrix(raw_frame)
        assert "quality" in matrix.columns
        assert set(FEATURE_NAMES).issubset(matrix.columns)

    def test_is_square_and_symmetric(self, raw_frame: pd.DataFrame) -> None:
        matrix = correlation_matrix(raw_frame)
        assert matrix.shape[0] == matrix.shape[1]
        assert np.allclose(matrix.values, matrix.values.T, equal_nan=True)

    def test_diagonal_is_one(self, raw_frame: pd.DataFrame) -> None:
        matrix = correlation_matrix(raw_frame)
        assert np.allclose(np.diag(matrix.values), 1.0)

    def test_alcohol_correlates_positively_with_quality(self, raw_frame: pd.DataFrame) -> None:
        """A documented domain fact — guards against a transposed or misaligned matrix."""
        matrix = correlation_matrix(raw_frame)
        assert matrix.loc["alcohol", "quality"] > 0.4

    def test_volatile_acidity_correlates_negatively_with_quality(
        self, raw_frame: pd.DataFrame
    ) -> None:
        matrix = correlation_matrix(raw_frame)
        assert matrix.loc["volatile acidity", "quality"] < -0.3