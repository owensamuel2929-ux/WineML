"""Tests for dataset loading, validation, and target derivation."""

from __future__ import annotations

import pandas as pd
import pytest

from wine_quality.config import FEATURE_NAMES, TARGET_NAME
from wine_quality.data.loader import (
    DataValidationError,
    describe_quality,
    load_and_profile,
    load_raw_data,
    profile_data,
)
from wine_quality.data.schema import build_targets, make_binary_target, split_features_target


class TestLoadRawData:
    """The loader must reproduce the dataset's documented shape exactly."""

    def test_loads_expected_shape(self, raw_frame: pd.DataFrame) -> None:
        assert raw_frame.shape == (1599, 12)

    def test_has_canonical_columns_in_order(self, raw_frame: pd.DataFrame) -> None:
        assert list(raw_frame.columns) == [*FEATURE_NAMES, TARGET_NAME]

    def test_contains_no_missing_values(self, raw_frame: pd.DataFrame) -> None:
        assert raw_frame.isna().sum().sum() == 0

    def test_all_columns_are_numeric(self, raw_frame: pd.DataFrame) -> None:
        assert all(pd.api.types.is_numeric_dtype(raw_frame[column]) for column in raw_frame.columns)

    def test_quality_within_documented_range(self, raw_frame: pd.DataFrame) -> None:
        assert raw_frame[TARGET_NAME].min() >= 3
        assert raw_frame[TARGET_NAME].max() <= 8

    def test_missing_file_raises_filenotfound(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
            load_raw_data(path=tmp_path / "does_not_exist.csv")

    def test_missing_column_raises_validation_error(self, tmp_path) -> None:
        broken = tmp_path / "broken.csv"
        pd.DataFrame({"fixed acidity": [7.4, 7.8], "quality": [5, 6]}).to_csv(broken, index=False)

        with pytest.raises(DataValidationError, match="missing required columns"):
            load_raw_data(path=broken)

    def test_unexpected_column_raises_validation_error(self, tmp_path) -> None:
        csv = tmp_path / "extra.csv"
        source = load_raw_data()
        source["mystery"] = 1.0
        source.to_csv(csv, index=False)

        with pytest.raises(DataValidationError, match="unexpected columns"):
            load_raw_data(path=csv)

    def test_out_of_range_values_raise_validation_error(self, tmp_path) -> None:
        corrupted = tmp_path / "corrupted.csv"
        source = load_raw_data()
        source.loc[0, "pH"] = 99.0  # impossible on the 0-14 scale
        source.to_csv(corrupted, index=False)

        with pytest.raises(DataValidationError, match="out-of-range"):
            load_raw_data(path=corrupted)

    def test_empty_file_raises_validation_error(self, tmp_path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text(",".join([*FEATURE_NAMES, TARGET_NAME]) + "\n", encoding="utf-8")

        with pytest.raises(DataValidationError):
            load_raw_data(path=empty)


class TestTargets:
    """Both framings must derive from the same pinned feature frame."""

    def test_binary_target_is_binary(self, raw_frame: pd.DataFrame) -> None:
        is_good = make_binary_target(raw_frame[TARGET_NAME])
        assert set(is_good.unique()) <= {0, 1}

    def test_binary_target_matches_cutoff(self, raw_frame: pd.DataFrame) -> None:
        is_good = make_binary_target(raw_frame[TARGET_NAME], cutoff=7)
        expected = (raw_frame[TARGET_NAME] >= 7).astype(int)
        pd.testing.assert_series_equal(is_good, expected.rename("is_good"))

    def test_custom_cutoff_changes_prevalence(self, raw_frame: pd.DataFrame) -> None:
        strict = make_binary_target(raw_frame[TARGET_NAME], cutoff=8).sum()
        lenient = make_binary_target(raw_frame[TARGET_NAME], cutoff=6).sum()
        assert strict < lenient

    def test_cutoff_of_seven_yields_minority_class(self, raw_frame: pd.DataFrame) -> None:
        """Reproduces the dataset card's ~13.6% positive rate."""
        rate = make_binary_target(raw_frame[TARGET_NAME], cutoff=7).mean()
        assert 0.10 < rate < 0.17, f"expected ~13.6% positive, got {rate:.1%}"

    def test_split_features_target_separates_columns(self, raw_frame: pd.DataFrame) -> None:
        features, quality = split_features_target(raw_frame)
        assert list(features.columns) == list(FEATURE_NAMES)
        assert TARGET_NAME not in features.columns
        assert quality.name == TARGET_NAME

    def test_build_targets_shares_one_feature_frame(self, raw_frame: pd.DataFrame) -> None:
        features, quality, is_good = build_targets(raw_frame)
        assert len(features) == len(quality) == len(is_good)
        assert list(features.columns) == list(FEATURE_NAMES)


class TestProfiling:
    """Profiling feeds the dashboard and the training provenance log."""

    def test_profile_reports_shape(self, raw_frame: pd.DataFrame) -> None:
        profile = profile_data(raw_frame)
        assert profile.n_rows == 1599
        assert profile.n_columns == 12
        assert profile.n_features == 11

    def test_profile_counts_duplicates(self, raw_frame: pd.DataFrame) -> None:
        profile = profile_data(raw_frame)
        assert profile.n_duplicate_rows == 240

    def test_positive_rate_and_imbalance(self, raw_frame: pd.DataFrame) -> None:
        profile = profile_data(raw_frame, cutoff=7)
        assert 0.10 < profile.positive_rate < 0.17
        assert profile.imbalance_ratio > 5.0

    def test_quality_distribution_includes_binary_count(self, raw_frame: pd.DataFrame) -> None:
        distribution = describe_quality(raw_frame, cutoff=7)
        assert -1 in distribution  # binary "good" count
        assert distribution[-1] == (raw_frame[TARGET_NAME] >= 7).sum()
        assert sum(count for score, count in distribution.items() if score > 0) == 1599

    def test_feature_summary_covers_every_feature(self, raw_frame: pd.DataFrame) -> None:
        profile = profile_data(raw_frame)
        assert set(profile.feature_summary.keys()) == set(FEATURE_NAMES)

    def test_load_and_profile_returns_both(self) -> None:
        frame, profile = load_and_profile()
        assert len(frame) == profile.n_rows == 1599