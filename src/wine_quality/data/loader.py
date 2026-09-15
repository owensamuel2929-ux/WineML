"""Loading and validation of the raw wine-quality dataset.

The dataset is small (1,599 rows) and static, so it is read from CSV on demand
rather than cached in a database. Validation runs on every load so that a
corrupted or replaced source file fails loudly at the start of a run instead of
silently degrading model quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from wine_quality.config import (
    FEATURE_BOUNDS,
    FEATURE_NAMES,
    TARGET_NAME,
    Settings,
    get_settings,
)


class DataValidationError(ValueError):
    """Raised when the source dataset violates its expected schema."""


@dataclass(frozen=True)
class DataProfile:
    """Summary statistics describing a loaded dataset.

    Returned alongside the frame so that training runs can log provenance and
    the dashboard can render an overview without re-reading the CSV.
    """

    n_rows: int
    n_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    n_duplicate_rows: int
    quality_distribution: dict[int, int]
    #: Threshold used to derive the "good wine" class.
    good_cutoff: int = 7
    feature_summary: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Number of predictor columns."""
        return len(self.columns) - 1

    @property
    def n_good_wines(self) -> int:
        """Rows scoring at or above :attr:`good_cutoff`."""
        return self.quality_distribution.get(-1, 0)

    @property
    def positive_rate(self) -> float:
        """Share of rows labelled "good" — the minority class prevalence."""
        if self.n_rows == 0:
            return 0.0
        return self.n_good_wines / self.n_rows

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of negative to positive examples (1.0 means perfectly balanced)."""
        positives = self.n_good_wines
        if positives == 0:
            return float("inf")
        return (self.n_rows - positives) / positives


def _validate_schema(frame: pd.DataFrame) -> None:
    """Assert the frame matches the expected column set and has no nulls."""
    missing = [column for column in (*FEATURE_NAMES, TARGET_NAME) if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")

    unexpected = [column for column in frame.columns if column not in (*FEATURE_NAMES, TARGET_NAME)]
    if unexpected:
        raise DataValidationError(f"Dataset contains unexpected columns: {unexpected}")

    null_counts = frame.isna().sum()
    if (nulls := null_counts[null_counts > 0]).any():
        raise DataValidationError(f"Dataset contains missing values: {nulls.to_dict()}")

    non_numeric = [
        column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DataValidationError(f"Dataset contains non-numeric columns: {non_numeric}")

    if frame.empty:
        raise DataValidationError("Dataset is empty")


def _validate_ranges(frame: pd.DataFrame) -> None:
    """Warn-free structural check that all values fall inside plausible bounds.

    Values outside the configured bounds indicate either a unit change or a
    corrupted file, both of which should stop a training run.
    """
    violations: dict[str, dict[str, float]] = {}
    for column, (low, high) in FEATURE_BOUNDS.items():
        if column not in frame.columns:
            continue
        observed_min = float(frame[column].min())
        observed_max = float(frame[column].max())
        if observed_min < low or observed_max > high:
            violations[column] = {
                "expected_min": low,
                "expected_max": high,
                "observed_min": observed_min,
                "observed_max": observed_max,
            }

    if violations:
        raise DataValidationError(f"Dataset contains out-of-range values: {violations}")


def load_raw_data(
    settings: Settings | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the raw CSV, validating its schema and value ranges.

    Args:
        settings: Settings instance providing default paths. Listed first
            because it is the common argument; passing it positionally must not
            silently bind to ``path``.
        path: Explicit path to the CSV, overriding the configured location.

    Returns:
        The validated dataframe with canonical column ordering.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        DataValidationError: If the schema or value ranges are violated.
    """
    settings = settings or get_settings()
    resolved = path or settings.raw_data_path

    if not resolved.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {resolved}. "
            "Download 'winequality-red.csv' from the Kaggle dataset into raw_data/."
        )

    frame = pd.read_csv(resolved)

    _validate_schema(frame)
    _validate_ranges(frame)

    # Enforce canonical ordering so downstream positional assumptions hold.
    return frame[[*FEATURE_NAMES, TARGET_NAME]]


def describe_quality(frame: pd.DataFrame, cutoff: int = 7) -> dict[int, int]:
    """Count rows per quality score.

    Args:
        frame: Validated dataset containing the target column.
        cutoff: Scores at or above this value are considered "good".

    Returns:
        Mapping of quality score to row count, including a ``-1`` key holding
        the count of "good" wines, so the full distribution and the binary
        split are available from a single call.
    """
    distribution = frame[TARGET_NAME].value_counts().sort_index().to_dict()
    counts = {int(score): int(count) for score, count in distribution.items()}
    counts[-1] = int((frame[TARGET_NAME] >= cutoff).sum())
    return counts


def profile_data(frame: pd.DataFrame, cutoff: int = 7) -> DataProfile:
    """Build a statistical profile of a validated dataset.

    Args:
        frame: Validated dataset.
        cutoff: Threshold defining a "good" wine.

    Returns:
        A :class:`DataProfile` describing shape, quality, and feature statistics.
    """
    feature_summary = {
        column: {
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std()),
            "min": float(frame[column].min()),
            "median": float(frame[column].median()),
            "max": float(frame[column].max()),
        }
        for column in FEATURE_NAMES
    }

    return DataProfile(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        columns=list(frame.columns),
        dtypes={column: str(dtype) for column, dtype in frame.dtypes.items()},
        missing_values={column: int(count) for column, count in frame.isna().sum().items()},
        n_duplicate_rows=int(frame.duplicated().sum()),
        quality_distribution=describe_quality(frame, cutoff=cutoff),
        good_cutoff=cutoff,
        feature_summary=feature_summary,
    )


def load_and_profile(settings: Settings | None = None) -> tuple[pd.DataFrame, DataProfile]:
    """Load the dataset and profile it in one step.

    Args:
        settings: Settings instance providing paths and the quality cutoff.

    Returns:
        Tuple of the validated dataframe and its profile.
    """
    settings = settings or get_settings()
    frame = load_raw_data(settings=settings)
    return frame, profile_data(frame, cutoff=settings.good_quality_cutoff)