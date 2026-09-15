"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wine_quality.config import FEATURE_NAMES, TARGET_NAME, Settings  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at the real dataset but writing artifacts to a temp dir.

    Keeps tests from clobbering the developer's trained models.
    """
    root = Path(__file__).resolve().parents[1]
    return Settings(
        project_root=root,
        models_dir=Path("models"),
        reports_dir=Path("reports"),
    )


@pytest.fixture(scope="session")
def isolated_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Session-scoped settings for the expensive end-to-end training test.

    Session scope lets one trained model be shared across all of that class's
    assertions instead of retraining for each one.
    """
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("trained")
    return Settings(
        project_root=root,
        models_dir=output / "models",
        reports_dir=output / "reports",
    )


@pytest.fixture
def fresh_settings(tmp_path: Path) -> Settings:
    """Function-scoped settings with an empty artifact directory.

    Used by tests that assert on the *absence* of artifacts or on run-log
    contents, which would otherwise be polluted by other tests sharing state.
    """
    root = Path(__file__).resolve().parents[1]
    return Settings(
        project_root=root,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture(scope="session")
def raw_frame():
    """The real dataset, loaded once per session."""
    from wine_quality.data.loader import load_raw_data

    return load_raw_data()


@pytest.fixture(scope="session")
def feature_names() -> tuple[str, ...]:
    """Canonical feature ordering."""
    return FEATURE_NAMES


@pytest.fixture(scope="session")
def target_name() -> str:
    """Canonical target column name."""
    return TARGET_NAME


@pytest.fixture(scope="session")
def sample_features() -> dict[str, float]:
    """A realistic feature payload for prediction tests."""
    return {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }