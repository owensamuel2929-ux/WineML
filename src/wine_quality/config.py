"""Centralised configuration for the wine-quality ML platform.

Every tunable value lives here so that no module hardcodes paths, thresholds,
or hyperparameters. Values can be overridden with environment variables using
the ``WINE_`` prefix (e.g. ``WINE_GOOD_QUALITY_CUTOFF=6``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Canonical dataset schema (Cortez et al., 2009)
# --------------------------------------------------------------------------
# Order matters: it defines the column order the API expects and the order the
# ColumnTransformer sees. Changing this list invalidates trained artifacts.
FEATURE_NAMES: tuple[str, ...] = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
)

TARGET_NAME: str = "quality"

#: Physicochemical properties are all continuous in this dataset.
NUMERIC_FEATURES: tuple[str, ...] = FEATURE_NAMES

#: Human-readable units/descriptions, surfaced in the API schema and dashboard.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "fixed acidity": "Tartaric acid concentration (g/dm³)",
    "volatile acidity": "Acetic acid concentration (g/dm³) — high values cause vinegar taste",
    "citric acid": "Citric acid concentration (g/dm³) — adds freshness",
    "residual sugar": "Sugar remaining after fermentation (g/dm³)",
    "chlorides": "Sodium chloride content (g/dm³) — saltiness",
    "free sulfur dioxide": "Free SO₂ (mg/dm³) — antimicrobial, prevents oxidation",
    "total sulfur dioxide": "Total SO₂ (mg/dm³)",
    "density": "Density (g/cm³) — close to that of water",
    "pH": "Acidity on the 0–14 scale (most wines 3.0–4.0)",
    "sulphates": "Potassium sulphate additive (g/dm³) — can contribute to SO₂ levels",
    "alcohol": "Alcohol content (% vol.)",
}

#: Plausible physical ranges, used for input validation. Derived from the
#: training distribution widened by a safety margin so the API rejects only
#: clearly impossible measurements rather than unusual-but-valid wines.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "fixed acidity": (2.0, 20.0),
    "volatile acidity": (0.0, 2.5),
    "citric acid": (0.0, 1.5),
    "residual sugar": (0.0, 30.0),
    "chlorides": (0.0, 1.0),
    "free sulfur dioxide": (0.0, 150.0),
    "total sulfur dioxide": (0.0, 400.0),
    "density": (0.98, 1.01),
    "pH": (2.5, 4.5),
    "sulphates": (0.0, 3.0),
    "alcohol": (7.0, 16.0),
}


class Settings(BaseSettings):
    """Runtime settings, sourced from environment variables where present."""

    model_config = SettingsConfigDict(
        env_prefix="WINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Paths ---------------------------------------------------------
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    raw_data_dir: Path = Field(default=Path("raw_data"))
    models_dir: Path = Field(default=Path("models"))
    reports_dir: Path = Field(default=Path("reports"))

    # -- Task definition -----------------------------------------------
    #: Sensory scores >= this value are labelled "good" (Kaggle's suggested cutoff).
    good_quality_cutoff: int = 7
    test_size: float = 0.2
    random_state: int = 42

    # -- Model selection -----------------------------------------------
    #: Candidates are evaluated with cross-validation; the best is persisted.
    cv_folds: int = 5
    #: Primary model-selection metric for the classifier. PR-AUC is preferred
    #: over ROC-AUC because "good" wines are a ~13.6% minority class.
    primary_metric: str = "average_precision"
    #: Handles the class imbalance without synthesising rows.
    class_weight: str | None = "balanced"

    # -- Serving --------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    #: Base URL the dashboard uses to reach the API. Overridden in compose.
    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: float = 10.0

    @field_validator("test_size")
    @classmethod
    def _validate_test_size(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("test_size must be strictly between 0 and 1")
        return value

    @field_validator("good_quality_cutoff")
    @classmethod
    def _validate_cutoff(cls, value: int) -> int:
        if not 3 <= value <= 9:
            raise ValueError("good_quality_cutoff must lie within the observed quality range 3-9")
        return value

    # -- Derived paths --------------------------------------------------
    @property
    def raw_data_path(self) -> Path:
        """Absolute path to the single source CSV."""
        return self.project_root / self.raw_data_dir / "winequality-red.csv"

    @property
    def classifier_path(self) -> Path:
        return self.project_root / self.models_dir / "classifier.joblib"

    @property
    def regressor_path(self) -> Path:
        return self.project_root / self.models_dir / "regressor.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.project_root / self.models_dir / "model_metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.project_root / self.reports_dir / "metrics.json"

    @property
    def runs_log_path(self) -> Path:
        """Append-only JSONL log of every training run."""
        return self.project_root / self.reports_dir / "runs.jsonl"

    def ensure_directories(self) -> None:
        """Create the output directories the pipeline writes to."""
        for directory in (self.models_dir, self.reports_dir):
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()