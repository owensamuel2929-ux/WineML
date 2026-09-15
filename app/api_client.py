"""HTTP client for the prediction API, shared by every dashboard page.

The dashboard deliberately does **not** import the models in-process. Going over
HTTP means the UI exercises the same interface any other client would, so a
broken API surfaces as a visible dashboard error instead of being masked by a
working local import.
"""

from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from wine_quality.config import Settings, get_settings


class APIError(RuntimeError):
    """Raised when the prediction API cannot be reached or returns an error."""


def _base_url(settings: Settings) -> str:
    """Resolve the API base URL, allowing a sidebar override for local debugging.

    Args:
        settings: Settings supplying the default URL.

    Returns:
        Base URL without a trailing slash.
    """
    override = st.session_state.get("api_base_url")
    return (override or settings.api_base_url).rstrip("/")


def _get(path: str, timeout: float) -> dict[str, Any]:
    """Issue a GET request against the API.

    Args:
        path: Path beginning with a slash.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response.

    Raises:
        APIError: On connection failure or a non-2xx response.
    """
    settings = get_settings()
    url = f"{_base_url(settings)}{path}"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.RequestError as exc:
        raise APIError(f"Cannot reach the prediction API at {url}: {exc}") from exc

    if response.status_code >= 400:
        raise APIError(f"API returned {response.status_code} for {path}: {response.text}")

    return response.json()


def _post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Issue a POST request against the API.

    Args:
        path: Path beginning with a slash.
        payload: JSON request body.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response.

    Raises:
        APIError: On connection failure or a non-2xx response.
    """
    settings = get_settings()
    url = f"{_base_url(settings)}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
    except httpx.RequestError as exc:
        raise APIError(f"Cannot reach the prediction API at {url}: {exc}") from exc

    if response.status_code == 422:
        detail = response.json().get("detail", response.text)
        raise APIError(f"Invalid input: {_format_validation_error(detail)}")

    if response.status_code >= 400:
        raise APIError(f"API returned {response.status_code} for {path}: {response.text}")

    return response.json()


def _format_validation_error(detail: Any) -> str:
    """Turn a FastAPI validation error into a readable sentence.

    Args:
        detail: The ``detail`` field from a 422 response.

    Returns:
        A human-readable summary of the offending fields.
    """
    if isinstance(detail, list):
        parts = []
        for item in detail:
            location = item.get("loc", [])
            field = location[-1] if location else "input"
            parts.append(f"{field}: {item.get('msg', 'invalid')}")
        return "; ".join(parts)
    return str(detail)


@st.cache_data(ttl=30, show_spinner=False)
def get_health() -> dict[str, Any]:
    """Fetch service health.

    Returns:
        Health payload with ``status`` and ``models_loaded``.
    """
    return _get("/health", timeout=get_settings().api_timeout_seconds)


@st.cache_data(ttl=60, show_spinner=False)
def get_model_info() -> dict[str, Any]:
    """Fetch served-model provenance and metrics.

    Returns:
        Model metadata payload.
    """
    return _get("/model-info", timeout=get_settings().api_timeout_seconds)


@st.cache_data(ttl=300, show_spinner=False)
def get_schema() -> dict[str, Any]:
    """Fetch the feature schema.

    Returns:
        Schema payload listing features, ranges, and descriptions.
    """
    return _get("/schema", timeout=get_settings().api_timeout_seconds)


def predict(features: dict[str, float]) -> dict[str, Any]:
    """Request predictions for a single wine sample.

    Not cached: predictions should reflect the latest submitted inputs.

    Args:
        features: Physicochemical measurements keyed by feature name.

    Returns:
        Combined classification and regression prediction payload.
    """
    payload = {name.replace(" ", "_"): value for name, value in features.items()}
    return _post("/predict", payload, timeout=get_settings().api_timeout_seconds)


def api_reachable() -> bool:
    """Check whether the API is up, for showing a friendly banner.

    Returns:
        ``True`` when ``/health`` responds successfully.
    """
    try:
        get_health()
    except APIError:
        return False
    return True