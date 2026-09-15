"""Pydantic request and response models for the prediction API.

Defining the wire contract explicitly means the dashboard, tests, and any other
client all speak the same schema, and FastAPI can publish an OpenAPI spec for
free. Feature bounds are pulled from the shared config so validation rules
cannot drift between the API and the training code.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from wine_quality.config import FEATURE_BOUNDS, FEATURE_DESCRIPTIONS, FEATURE_NAMES

#: Example payload mirroring a typical mid-range red wine, used in the docs.
EXAMPLE_WINE: dict[str, float] = {
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


def _build_feature_fields() -> dict[str, Any]:
    """Generate one constrained Pydantic field per physicochemical property.

    Returns:
        Mapping of field name to ``(annotation, FieldInfo)`` for
        :func:`pydantic.create_model`. Building the model dynamically keeps the
        schema and the config's bounds in lockstep — adding a feature to
        ``FEATURE_NAMES`` cannot silently omit it from the API.
    """
    fields: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        low, high = FEATURE_BOUNDS[name]
        fields[name.replace(" ", "_")] = (
            Annotated[float, Field(ge=low, le=high, description=FEATURE_DESCRIPTIONS[name])],
            ...,
        )
    return fields


#: Dynamically generated request body: one required, range-validated field per
#: physicochemical property.
_WineFeaturesBase = create_model(
    "WineFeatures",
    __config__=ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_WINE},
    ),
    **_build_feature_fields(),
)


class WineFeatures(_WineFeaturesBase):  # type: ignore[misc,valid-type]
    """Physicochemical measurements for a single wine sample."""

    def to_feature_dict(self) -> dict[str, float]:
        """Convert to the canonical feature mapping used by the pipelines.

        Returns:
            Mapping keyed by original column names (with spaces), matching
            :data:`~wine_quality.config.FEATURE_NAMES`.
        """
        return {name: float(getattr(self, name.replace(" ", "_"))) for name in FEATURE_NAMES}


class ClassificationPrediction(BaseModel):
    """Output of the "is this a good wine?" task."""

    is_good: bool = Field(description="True when the predicted probability meets the threshold")
    probability_good: float = Field(
        ge=0.0, le=1.0, description="Model probability that quality >= cutoff"
    )
    threshold: float = Field(description="Decision threshold used to produce `is_good`")
    model_version: str = Field(description="Version of the serving model")


class RegressionPrediction(BaseModel):
    """Output of the sensory-score task."""

    predicted_quality: float = Field(description="Predicted quality score on the 0-10 scale")
    rounded_quality: int = Field(description="Predicted score rounded to the nearest integer")
    model_version: str = Field(description="Version of the serving model")


class PredictionResponse(BaseModel):
    """Combined response returning both tasks for one sample.

    The dashboard displays both together because they answer different
    questions: the score places the wine on the original scale, while the
    probability expresses confidence that it clears the "good" threshold.
    """

    classification: ClassificationPrediction
    regression: RegressionPrediction
    input_features: dict[str, float] = Field(
        description="Echo of the validated input, for traceability"
    )


class FeatureRange(BaseModel):
    """Plausible measurement bounds for one feature."""

    low: float
    high: float
    description: str


class HealthResponse(BaseModel):
    """Liveness and readiness status of the service."""

    status: str = Field(description="'ok' when the service is ready to serve")
    models_loaded: bool = Field(description="Whether model artifacts were loaded successfully")


class ModelInfoResponse(BaseModel):
    """Description of the currently served model, read from artifact metadata.

    Every field the dashboard renders must appear here. Pydantic response
    models filter out anything not declared, so an omission here surfaces as a
    ``KeyError`` in the Streamlit page rather than an error at the API boundary.
    """

    model_version: str
    trained_at: str
    good_quality_cutoff: int
    feature_names: list[str]
    n_training_rows: int = Field(description="Rows used to fit the served models")
    n_test_rows: int = Field(description="Rows held out for evaluation")
    classifier_algorithm: str
    regressor_algorithm: str
    classification_metrics: dict[str, Any]
    regression_metrics: dict[str, Any]
    sklearn_version: str
    python_version: str


class SchemaResponse(BaseModel):
    """Feature schema, letting dashboard widgets configure themselves."""

    feature_names: list[str]
    feature_ranges: dict[str, FeatureRange]
    feature_descriptions: dict[str, str]
    good_quality_cutoff: int