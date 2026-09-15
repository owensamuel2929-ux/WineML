"""FastAPI application exposing the trained wine-quality models.

The service loads both pipelines once at startup and keeps them in memory.
Predictions run through the *serialised* pipelines, so the preprocessing applied
here is byte-identical to the preprocessing used during training.

Run locally::

    uvicorn wine_quality.serving.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status

from wine_quality.config import FEATURE_DESCRIPTIONS, FEATURE_NAMES, Settings, get_settings
from wine_quality.features.pipeline import FeatureValidationError, feature_ranges, to_frame
from wine_quality.models.registry import ArtifactError, ModelMetadata, load_artifacts
from wine_quality.serving.schemas import (
    ClassificationPrediction,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RegressionPrediction,
    SchemaResponse,
    WineFeatures,
)

logger = logging.getLogger(__name__)

#: Decision threshold on P(good). Left at 0.5 because the classifier was trained
#: with balanced class weights, so the model's probabilities are already
#: adjusted for the ~13.6% base rate. Lowering it trades precision for recall.
DECISION_THRESHOLD: float = 0.5


class ModelStore:
    """In-memory holder for the loaded pipelines and their metadata.

    A tiny explicit container rather than module-level globals, so tests can
    construct an isolated store instead of mutating process state.
    """

    def __init__(self) -> None:
        self.classifier: Any = None
        self.regressor: Any = None
        self.metadata: ModelMetadata | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether both pipelines and their metadata are available."""
        return (
            self.classifier is not None
            and self.regressor is not None
            and self.metadata is not None
        )

    def load(self, settings: Settings) -> None:
        """Load artifacts from disk into memory.

        Args:
            settings: Settings supplying artifact paths.

        Raises:
            ArtifactError: If the artifacts are missing or unreadable.
        """
        self.classifier, self.regressor, self.metadata = load_artifacts(settings)

        # Force single-threaded inference. Tree ensembles average their
        # predictions with a parallel reduction, and floating-point addition is
        # not associative, so `n_jobs=-1` makes repeated identical requests
        # differ in the last bit (~1e-16). Training benefits from parallelism;
        # scoring one row does not, and reproducible predictions matter for
        # caching, auditing, and debugging.
        for pipeline in (self.classifier, self.regressor):
            estimator = pipeline.named_steps.get("model")
            if estimator is not None and hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 1

        logger.info(
            "Loaded model version %s (classifier=%s, regressor=%s)",
            self.metadata.model_version,
            self.metadata.classifier_algorithm,
            self.metadata.regressor_algorithm,
        )

    def clear(self) -> None:
        """Drop the loaded models, used by tests between cases."""
        self.classifier = None
        self.regressor = None
        self.metadata = None


#: Process-wide model store.
store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, tolerating their absence.

    Serving without artifacts is a hard failure for predictions but a soft one
    for the process: the container still starts so ``/health`` can report the
    problem instead of crash-looping, which makes the failure much easier to
    diagnose from `docker compose ps`.
    """
    settings = get_settings()
    try:
        store.load(settings)
    except ArtifactError as exc:
        logger.warning("Starting without model artifacts: %s", exc)

    yield

    store.clear()


app = FastAPI(
    title="Wine Quality Prediction API",
    description=(
        "Predicts red wine quality from physicochemical properties, following "
        "Cortez et al. (2009). Exposes both the binary 'good wine' "
        "classification task and the sensory-score regression task."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _require_models() -> ModelMetadata:
    """Assert models are loaded, returning their metadata.

    Returns:
        Metadata for the loaded model.

    Raises:
        HTTPException: 503 if artifacts were not loaded at startup.
    """
    if not store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model artifacts are not loaded. Run the training stage first "
                "(`make train` or `docker compose --profile train up`)."
            ),
        )
    assert store.metadata is not None  # narrowed by is_loaded
    return store.metadata


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report service liveness and model readiness."""
    return HealthResponse(
        status="ok" if store.is_loaded else "degraded",
        models_loaded=store.is_loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Describe the served model: algorithms, provenance, and held-out metrics."""
    metadata = _require_models()
    return ModelInfoResponse(**metadata.to_dict())


@app.get("/schema", response_model=SchemaResponse, tags=["ops"])
def schema() -> SchemaResponse:
    """Return the feature schema so clients can build input forms dynamically."""
    return SchemaResponse(
        feature_names=list(FEATURE_NAMES),
        feature_ranges=feature_ranges(),
        feature_descriptions=dict(FEATURE_DESCRIPTIONS),
        good_quality_cutoff=get_settings().good_quality_cutoff,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: WineFeatures) -> PredictionResponse:
    """Predict both the quality score and the probability of a good wine.

    Args:
        features: Validated physicochemical measurements.

    Returns:
        Combined classification and regression predictions.

    Raises:
        HTTPException: 422 for invalid features, 503 if models are unavailable.
    """
    metadata = _require_models()
    payload = features.to_feature_dict()

    try:
        frame = to_frame(payload)
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    probability_good = float(store.classifier.predict_proba(frame)[0, 1])
    predicted_quality = float(store.regressor.predict(frame)[0])

    # The regressor is unconstrained, so clamp to the scale the dataset defines.
    predicted_quality = float(np.clip(predicted_quality, 0.0, 10.0))

    return PredictionResponse(
        classification=ClassificationPrediction(
            is_good=probability_good >= DECISION_THRESHOLD,
            probability_good=probability_good,
            threshold=DECISION_THRESHOLD,
            model_version=metadata.model_version,
        ),
        regression=RegressionPrediction(
            predicted_quality=predicted_quality,
            rounded_quality=int(round(predicted_quality)),
            model_version=metadata.model_version,
        ),
        input_features=payload,
    )


@app.post("/predict/classification", response_model=ClassificationPrediction, tags=["prediction"])
def predict_classification(features: WineFeatures) -> ClassificationPrediction:
    """Predict only the binary "good wine" outcome.

    Args:
        features: Validated physicochemical measurements.

    Returns:
        The classification verdict and its probability.
    """
    return predict(features).classification


@app.post("/predict/regression", response_model=RegressionPrediction, tags=["prediction"])
def predict_regression(features: WineFeatures) -> RegressionPrediction:
    """Predict only the continuous sensory score.

    Args:
        features: Validated physicochemical measurements.

    Returns:
        The predicted quality score.
    """
    return predict(features).regression


@app.post("/reload", response_model=HealthResponse, tags=["ops"])
def reload_models() -> HealthResponse:
    """Re-read artifacts from disk without restarting the container.

    Useful after a training run completes, so the serving container can pick up
    a new model without a full redeploy.

    Returns:
        The resulting health status.
    """
    settings = get_settings()
    try:
        store.load(settings)
    except ArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return HealthResponse(status="ok", models_loaded=True)