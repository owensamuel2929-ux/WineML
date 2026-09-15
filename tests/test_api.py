"""Tests for the prediction API contract.

Uses FastAPI's TestClient against a model store loaded from the real artifacts.
Tests that need artifacts are skipped with a clear message when training has not
been run, so the suite stays green on a fresh clone.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wine_quality.config import FEATURE_NAMES
from wine_quality.models.registry import artifacts_exist
from wine_quality.serving.api import app, store

pytestmark = pytest.mark.skipif(
    not artifacts_exist(),
    reason="Model artifacts not found — run `make train` before the API tests.",
)


@pytest.fixture(scope="module")
def client():
    """A TestClient with the lifespan startup executed, so models are loaded."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _ensure_models_loaded(client):
    """Guarantee the store is populated for every test in this module."""
    if not store.is_loaded:
        pytest.skip("Model artifacts could not be loaded.")


class TestHealth:
    """Health must distinguish 'process up' from 'model ready'."""

    def test_health_returns_ok(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["models_loaded"] is True


class TestModelInfo:
    """The API must describe the model it is actually serving."""

    def test_returns_provenance(self, client) -> None:
        response = client.get("/model-info")
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_version"]
        assert payload["trained_at"]
        assert payload["good_quality_cutoff"] == 7
        assert payload["feature_names"] == list(FEATURE_NAMES)

    def test_reports_both_algorithms(self, client) -> None:
        payload = client.get("/model-info").json()
        assert payload["classifier_algorithm"]
        assert payload["regressor_algorithm"]

    def test_includes_held_out_metrics(self, client) -> None:
        payload = client.get("/model-info").json()
        assert 0.0 <= payload["classification_metrics"]["roc_auc"] <= 1.0
        assert payload["regression_metrics"]["rmse"] > 0.0

    def test_exposes_row_counts(self, client) -> None:
        """Regression guard: the dashboard renders these row counts.

        Pydantic response models silently drop undeclared fields, so omitting a
        field here produces a ``KeyError`` in the Streamlit page rather than an
        error at the API boundary. Assert the full set the dashboard relies on.
        """
        payload = client.get("/model-info").json()
        assert payload["n_training_rows"] > 0
        assert payload["n_test_rows"] > 0
        assert payload["n_training_rows"] + payload["n_test_rows"] == 1599

    def test_exposes_every_field_the_dashboard_renders(self, client) -> None:
        """Pin the contract between /model-info and the Overview page."""
        required = {
            "model_version",
            "trained_at",
            "good_quality_cutoff",
            "classifier_algorithm",
            "regressor_algorithm",
            "sklearn_version",
            "n_training_rows",
            "n_test_rows",
        }
        payload = client.get("/model-info").json()
        assert required <= set(payload), f"missing: {required - set(payload)}"


class TestSchema:
    """The schema endpoint drives the dashboard's input form."""

    def test_lists_every_feature(self, client) -> None:
        payload = client.get("/schema").json()
        assert payload["feature_names"] == list(FEATURE_NAMES)

    def test_provides_ranges_for_every_feature(self, client) -> None:
        payload = client.get("/schema").json()
        assert set(payload["feature_ranges"].keys()) == set(FEATURE_NAMES)
        for bounds in payload["feature_ranges"].values():
            assert bounds["low"] < bounds["high"]

    def test_provides_descriptions(self, client) -> None:
        payload = client.get("/schema").json()
        assert all(payload["feature_descriptions"][name] for name in FEATURE_NAMES)


class TestPredict:
    """The core prediction contract."""

    def test_returns_both_tasks(self, client, sample_features: dict[str, float]) -> None:
        response = client.post(
            "/predict",
            json={name.replace(" ", "_"): value for name, value in sample_features.items()},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "classification" in payload
        assert "regression" in payload

    def test_classification_shape(self, client, sample_features: dict[str, float]) -> None:
        payload = client.post(
            "/predict",
            json={name.replace(" ", "_"): value for name, value in sample_features.items()},
        ).json()
        classification = payload["classification"]
        assert isinstance(classification["is_good"], bool)
        assert 0.0 <= classification["probability_good"] <= 1.0
        assert classification["threshold"] == 0.5

    def test_regression_shape(self, client, sample_features: dict[str, float]) -> None:
        payload = client.post(
            "/predict",
            json={name.replace(" ", "_"): value for name, value in sample_features.items()},
        ).json()
        regression = payload["regression"]
        assert 0.0 <= regression["predicted_quality"] <= 10.0
        assert isinstance(regression["rounded_quality"], int)

    def test_echoes_validated_input(self, client, sample_features: dict[str, float]) -> None:
        payload = client.post(
            "/predict",
            json={name.replace(" ", "_"): value for name, value in sample_features.items()},
        ).json()
        assert payload["input_features"] == sample_features

    def test_is_deterministic(self, client, sample_features: dict[str, float]) -> None:
        body = {name.replace(" ", "_"): value for name, value in sample_features.items()}
        first = client.post("/predict", json=body).json()
        second = client.post("/predict", json=body).json()
        assert first == second

    def test_high_alcohol_scores_higher_than_low(self, client, sample_features: dict[str, float]) -> None:
        """A directional sanity check: alcohol is the strongest known signal."""
        low = {**sample_features, "alcohol": 8.5}
        high = {**sample_features, "alcohol": 13.5}

        low_payload = client.post(
            "/predict", json={name.replace(" ", "_"): value for name, value in low.items()}
        ).json()
        high_payload = client.post(
            "/predict", json={name.replace(" ", "_"): value for name, value in high.items()}
        ).json()

        assert (
            high_payload["classification"]["probability_good"]
            > low_payload["classification"]["probability_good"]
        )
        assert (
            high_payload["regression"]["predicted_quality"]
            > low_payload["regression"]["predicted_quality"]
        )

    def test_rejects_missing_feature(self, client, sample_features: dict[str, float]) -> None:
        incomplete = {
            name.replace(" ", "_"): value
            for name, value in sample_features.items()
            if name != "alcohol"
        }
        assert client.post("/predict", json=incomplete).status_code == 422

    def test_rejects_out_of_range_value(self, client, sample_features: dict[str, float]) -> None:
        payload = {name.replace(" ", "_"): value for name, value in sample_features.items()}
        payload["pH"] = 99.0
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_unknown_field(self, client, sample_features: dict[str, float]) -> None:
        payload = {name.replace(" ", "_"): value for name, value in sample_features.items()}
        payload["grape_variety"] = 1.0
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_non_numeric_value(self, client, sample_features: dict[str, float]) -> None:
        payload = {name.replace(" ", "_"): value for name, value in sample_features.items()}
        payload["alcohol"] = "strong"
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_empty_body(self, client) -> None:
        assert client.post("/predict", json={}).status_code == 422


class TestSingleTaskEndpoints:
    """The convenience endpoints must agree with the combined one.

    Floats are compared with a tolerance rather than for exact equality: the
    combined response nests the value inside another model, and pydantic's JSON
    encoder can differ from the top-level path by a single ULP (~1e-16). That is
    serialisation noise, not a disagreement about the prediction.
    """

    def test_classification_endpoint_matches_combined(
        self, client, sample_features: dict[str, float]
    ) -> None:
        body = {name.replace(" ", "_"): value for name, value in sample_features.items()}
        combined = client.post("/predict", json=body).json()["classification"]
        single = client.post("/predict/classification", json=body).json()

        assert combined["is_good"] == single["is_good"]
        assert combined["threshold"] == single["threshold"]
        assert combined["model_version"] == single["model_version"]
        assert combined["probability_good"] == pytest.approx(single["probability_good"], abs=1e-12)

    def test_regression_endpoint_matches_combined(
        self, client, sample_features: dict[str, float]
    ) -> None:
        body = {name.replace(" ", "_"): value for name, value in sample_features.items()}
        combined = client.post("/predict", json=body).json()["regression"]
        single = client.post("/predict/regression", json=body).json()

        assert combined["rounded_quality"] == single["rounded_quality"]
        assert combined["model_version"] == single["model_version"]
        assert combined["predicted_quality"] == pytest.approx(
            single["predicted_quality"], abs=1e-12
        )


class TestReload:
    """Reload lets a new model be picked up without a redeploy."""

    def test_reload_succeeds(self, client) -> None:
        response = client.post("/reload")
        assert response.status_code == 200
        assert response.json()["models_loaded"] is True


class TestOpenAPI:
    """The generated spec must document every endpoint."""

    def test_spec_is_available(self, client) -> None:
        payload = client.get("/openapi.json").json()
        assert payload["info"]["title"] == "Wine Quality Prediction API"

    def test_all_endpoints_documented(self, client) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path in (
            "/health",
            "/model-info",
            "/schema",
            "/predict",
            "/predict/classification",
            "/predict/regression",
            "/reload",
        ):
            assert path in paths, f"{path} missing from the OpenAPI spec"