"""Persistence and retrieval of trained model artifacts.

Two artifacts are produced from one training run:

* ``classifier.joblib`` — pipeline predicting P(good wine).
* ``regressor.joblib``  — pipeline predicting the continuous sensory score.
* ``model_metadata.json`` — the accompanying contract: feature order, cutoff,
  metrics, library versions, and timestamp.

The metadata file is what the API reads to describe itself, so a model binary
can never be served without its provenance being available.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import sklearn

from wine_quality.config import FEATURE_NAMES, Settings, get_settings


class ArtifactError(RuntimeError):
    """Raised when a model artifact is missing or unreadable."""


@dataclass
class ModelMetadata:
    """Provenance and performance record for a training run."""

    model_version: str
    trained_at: str
    good_quality_cutoff: int
    feature_names: list[str]
    n_training_rows: int
    n_test_rows: int
    classifier_algorithm: str
    regressor_algorithm: str
    classification_metrics: dict[str, Any] = field(default_factory=dict)
    regression_metrics: dict[str, Any] = field(default_factory=dict)
    sklearn_version: str = sklearn.__version__
    python_version: str = platform.python_version()

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC timestamp in ISO-8601 form."""
        return datetime.now(UTC).isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping."""
        return asdict(self)


def save_artifacts(
    classifier: Any,
    regressor: Any,
    metadata: ModelMetadata,
    settings: Settings | None = None,
) -> dict[str, str]:
    """Persist both pipelines and their metadata to disk.

    Args:
        classifier: Fitted classification pipeline.
        regressor: Fitted regression pipeline.
        metadata: Provenance record for the run.
        settings: Settings supplying output paths.

    Returns:
        Mapping of artifact name to the path it was written to.
    """
    settings = settings or get_settings()
    settings.ensure_directories()

    joblib.dump(classifier, settings.classifier_path)
    joblib.dump(regressor, settings.regressor_path)

    settings.metadata_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "classifier": str(settings.classifier_path),
        "regressor": str(settings.regressor_path),
        "metadata": str(settings.metadata_path),
    }


def load_artifacts(settings: Settings | None = None) -> tuple[Any, Any, ModelMetadata]:
    """Load both pipelines and their metadata.

    Args:
        settings: Settings supplying artifact paths.

    Returns:
        Tuple of ``(classifier, regressor, metadata)``.

    Raises:
        ArtifactError: If any required artifact is missing.
    """
    settings = settings or get_settings()

    missing = [
        path.name
        for path in (settings.classifier_path, settings.regressor_path, settings.metadata_path)
        if not path.exists()
    ]
    if missing:
        raise ArtifactError(
            f"Missing model artifacts: {missing}. Run `make train` (or the compose "
            "training profile) to produce them."
        )

    classifier = joblib.load(settings.classifier_path)
    regressor = joblib.load(settings.regressor_path)
    metadata = ModelMetadata(**json.loads(settings.metadata_path.read_text(encoding="utf-8")))

    return classifier, regressor, metadata


def artifacts_exist(settings: Settings | None = None) -> bool:
    """Check whether a complete artifact set is present.

    Args:
        settings: Settings supplying artifact paths.

    Returns:
        ``True`` if all three artifacts exist.
    """
    settings = settings or get_settings()
    return all(
        path.exists()
        for path in (
            settings.classifier_path,
            settings.regressor_path,
            settings.metadata_path,
        )
    )


def append_run_log(record: dict[str, Any], settings: Settings | None = None) -> None:
    """Append a training run record to the JSONL experiment log.

    A deliberately minimal substitute for a tracking server: the dataset is tiny
    and runs are infrequent, so a newline-delimited file that ``git diff`` can
    read is more useful than operational overhead.

    Args:
        record: JSON-serialisable run summary.
        settings: Settings supplying the log path.
    """
    settings = settings or get_settings()
    settings.ensure_directories()

    with settings.runs_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_run_log(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Read every record from the JSONL experiment log.

    Args:
        settings: Settings supplying the log path.

    Returns:
        List of run records, oldest first. Empty if no runs have been logged.
    """
    settings = settings or get_settings()
    if not settings.runs_log_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in settings.runs_log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_metadata(
    classifier: Any,
    regressor: Any,
    n_training_rows: int,
    n_test_rows: int,
    classification_metrics: dict[str, Any],
    regression_metrics: dict[str, Any],
    cutoff: int,
) -> ModelMetadata:
    """Assemble the metadata record for a completed run.

    Args:
        classifier: Fitted classification pipeline.
        regressor: Fitted regression pipeline.
        n_training_rows: Number of rows used for fitting.
        n_test_rows: Number of rows held out for evaluation.
        classification_metrics: Metrics from the held-out set.
        regression_metrics: Metrics from the held-out set.
        cutoff: Threshold used to define the positive class.

    Returns:
        A populated :class:`ModelMetadata`.
    """
    return ModelMetadata(
        model_version="0.1.0",
        trained_at=ModelMetadata.now_iso(),
        good_quality_cutoff=cutoff,
        feature_names=list(FEATURE_NAMES),
        n_training_rows=n_training_rows,
        n_test_rows=n_test_rows,
        classifier_algorithm=type(classifier.named_steps["model"]).__name__,
        regressor_algorithm=type(regressor.named_steps["model"]).__name__,
        classification_metrics=classification_metrics,
        regression_metrics=regression_metrics,
    )