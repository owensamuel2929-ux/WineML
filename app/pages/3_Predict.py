"""Interactive prediction page.

Collects physicochemical measurements, sends them to the prediction API over
HTTP, and presents both the verdict and the score. The input widgets are built
from the API's own ``/schema`` endpoint, so the form cannot drift out of sync
with what the model expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit runs each page as a standalone script, so neither the project root
# nor `src` is on sys.path. This file lives in app/pages/, hence parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from app.api_client import APIError, get_schema, predict  # noqa: E402

from wine_quality.config import FEATURE_NAMES, get_settings  # noqa: E402
from wine_quality.data.loader import load_raw_data  # noqa: E402

st.set_page_config(page_title="Wine Quality · Predict", page_icon="🎯", layout="wide")

settings = get_settings()
st.title("🎯 Predict Wine Quality")
st.caption(
    "Adjust the physicochemical measurements and get both a quality score and the "
    "probability that the wine clears the 'good' threshold."
)

# ---------------------------------------------------------------------------
# Schema drives the form
# ---------------------------------------------------------------------------
try:
    schema = get_schema()
except APIError as exc:
    st.error(f"Prediction API unavailable: {exc}")
    st.info(
        "Start the API, then reload. The input form is generated from `/schema` so it "
        "always matches the model's expectations."
    )
    st.stop()

ranges = schema["feature_ranges"]
descriptions = schema["feature_descriptions"]
cutoff = schema["good_quality_cutoff"]
features = schema["feature_names"]

# ---------------------------------------------------------------------------
# Presets from real data — median and a few archetypes
# ---------------------------------------------------------------------------
try:
    raw = load_raw_data(settings)
    PRESETS = {
        "Median wine": raw[list(FEATURE_NAMES)].median().to_dict(),
        "Typical 'good' wine": raw[raw["quality"] >= cutoff][list(FEATURE_NAMES)].median().to_dict(),
        "Typical ordinary wine": raw[raw["quality"] < cutoff][list(FEATURE_NAMES)].median().to_dict(),
        "Highest-scoring wine": raw.loc[raw["quality"].idxmax(), list(FEATURE_NAMES)].to_dict(),
    }
except (FileNotFoundError, ValueError):
    PRESETS = {}

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Measurements")

preset_columns = st.columns([2, 1])
with preset_columns[0]:
    preset_name = st.selectbox(
        "Start from a real wine profile",
        options=list(PRESETS.keys()) if PRESETS else ["Custom"],
        help="Presets are medians computed from the dataset, so they are realistic starting points.",
    )
with preset_columns[1]:
    st.write("")
    if st.button("Reset to defaults", width="stretch"):
        for feature in features:
            st.session_state.pop(f"input_{feature}", None)
        st.rerun()

selected_preset = PRESETS.get(preset_name, {})

# Two columns of sliders keeps the whole form visible without scrolling.
left_column, right_column = st.columns(2)
values: dict[str, float] = {}

for index, feature in enumerate(features):
    target_column = left_column if index % 2 == 0 else right_column
    low = ranges[feature]["low"]
    high = ranges[feature]["high"]

    # Default to the preset value, clamped into the allowed range.
    default = float(selected_preset.get(feature, (low + high) / 2))
    default = min(max(default, low), high)
    step = (high - low) / 200

    with target_column:
        values[feature] = st.slider(
            feature,
            min_value=float(low),
            max_value=float(high),
            value=default,
            step=float(step),
            help=descriptions.get(feature, ""),
            key=f"input_{feature}",
        )

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.divider()

if not st.button("Predict quality", type="primary", width="stretch"):
    st.caption("Adjust the sliders and press **Predict quality**.")
    st.stop()

try:
    result = predict(values)
except APIError as exc:
    st.error(str(exc))
    st.stop()

classification = result["classification"]
regression = result["regression"]
probability = classification["probability_good"]
predicted = regression["predicted_quality"]
is_good = classification["is_good"]

st.subheader("Result")

verdict_column, score_column, confidence_column = st.columns(3)

with verdict_column:
    if is_good:
        st.success(f"### ✅ Good wine\nQuality ≥ {cutoff} predicted")
    else:
        st.info(f"### ➖ Ordinary wine\nQuality < {cutoff} predicted")

with score_column:
    st.metric(
        "Predicted quality",
        f"{predicted:.2f}",
        delta=f"rounded: {regression['rounded_quality']}",
        delta_color="off",
    )

with confidence_column:
    st.metric(
        "P(good wine)",
        f"{probability:.1%}",
        delta=f"threshold: {classification['threshold']:.0%}",
        delta_color="off",
    )

# -- Gauge ---------------------------------------------------------------
gauge_left, gauge_right = st.columns(2)

with gauge_left:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Probability of being a good wine"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#722F37"},
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.8,
                    "value": classification["threshold"] * 100,
                },
                "steps": [
                    {"range": [0, 50], "color": "#F2E8E7"},
                    {"range": [50, 100], "color": "#E8D3D0"},
                ],
            },
        )
    )
    figure.update_layout(height=300, margin=dict(t=50, b=10))
    st.plotly_chart(figure, width="stretch")

with gauge_right:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted,
            number={"font": {"size": 34}},
            title={"text": "Predicted quality score"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#722F37"},
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.8,
                    "value": cutoff,
                },
                "steps": [
                    {"range": [0, 3], "color": "#F2E8E7"},
                    {"range": [3, 6], "color": "#EBD9D6"},
                    {"range": [6, 8], "color": "#E0C4C0"},
                    {"range": [8, 10], "color": "#D5AFAA"},
                ],
            },
        )
    )
    figure.update_layout(height=300, margin=dict(t=50, b=10))
    st.plotly_chart(figure, width="stretch")

# -- Interpretation ------------------------------------------------------
with st.expander("How to read this result", expanded=False):
    st.markdown(
        f"""
**The two numbers answer different questions.**

- The **quality score** ({predicted:.2f}) places the wine on the original 0–10 sensory
  scale. It comes from the regressor, which was trained directly on the tasting scores.
- The **probability** ({probability:.1%}) comes from the classifier and expresses
  confidence that the wine clears the cutoff of {cutoff}. The classifier is trained with
  balanced class weights, so this probability is already adjusted for the fact that only
  ~13.6% of wines are "good" — it is not inflated by the majority class.

**They can disagree.** A wine with a predicted score of {cutoff - 0.4:.1f} might still
show a probability above 50%, because the two models are fitted separately and optimise
different objectives. Read the score for ranking and the probability for confidence.

**Regression to the mean is expected.** The regressor rarely predicts extreme scores,
because very few wines in the dataset score 3 or 8. Predictions cluster in the 5–6 range,
which reflects genuine uncertainty rather than a flaw.

**These are correlations, not causes.** The model learns statistical associations in the
training data. The dataset card explicitly warns against reading causal claims into the
physicochemical measures.
"""
    )

# -- Contribution breakdown ---------------------------------------------
with st.expander("Feature contributions to this prediction", expanded=False):
    st.caption(
        "Each feature's value shown against the training distribution, so you can see "
        "which measurements are unusual for this wine."
    )
    try:
        raw = load_raw_data(settings)
        comparison = pd.DataFrame(
            {
                "Feature": features,
                "Your value": [values[feature] for feature in features],
                "Dataset median": [float(raw[feature].median()) for feature in features],
                "Dataset 5th pct": [float(raw[feature].quantile(0.05)) for feature in features],
                "Dataset 95th pct": [float(raw[feature].quantile(0.95)) for feature in features],
            }
        )
        comparison["Outside 5–95% range"] = [
            "⚠️ yes"
            if (values[feature] < comparison.loc[index, "Dataset 5th pct"]
                or values[feature] > comparison.loc[index, "Dataset 95th pct"])
            else "no"
            for index, feature in enumerate(features)
        ]
        st.dataframe(
            comparison.style.format(
                {
                    "Your value": "{:.4f}",
                    "Dataset median": "{:.4f}",
                    "Dataset 5th pct": "{:.4f}",
                    "Dataset 95th pct": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Values outside the 5th–95th percentile are still valid inputs — the model "
            "handles them, but they are extrapolations relative to its training data, so "
            "predictions for such wines are less reliable."
        )
    except (FileNotFoundError, ValueError) as exc:
        st.caption(f"Could not load the dataset for comparison: {exc}")

st.caption(f"Served by model version `{classification['model_version']}`.")