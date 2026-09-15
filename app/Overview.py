"""Summary landing page for the wine-quality dashboard.

Shows dataset shape, the class-imbalance story, the served model's provenance,
and its held-out performance. Charts read the raw CSV directly (static data),
while anything about the *model* is fetched from the API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Streamlit runs this file as a script, so neither the project root nor `src`
# is on sys.path. Both are needed: the root to import `app.api_client`, and
# `src` to import the `wine_quality` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from app.api_client import APIError, get_health, get_model_info  # noqa: E402

from wine_quality.config import TARGET_NAME, get_settings  # noqa: E402
from wine_quality.data.loader import load_and_profile  # noqa: E402

st.set_page_config(page_title="Wine Quality · Overview", page_icon="🍷", layout="wide")

settings = get_settings()
st.title("🍷 Red Wine Quality")
st.caption(
    "Predicting perceived quality from physicochemical tests — "
    "Cortez et al. (2009), *Decision Support Systems* 47(4):547–553."
)


def _load_importances() -> dict[str, float] | None:
    """Read feature importances from the reports directory.

    Kept separate from the API because importances are run artifacts rather than
    something the serving container needs to expose.

    Returns:
        Feature importance mapping, or ``None`` if no report exists yet.
    """
    path = settings.metrics_path
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("feature_importance")


# ---------------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------------
models_ready = False
try:
    health = get_health()
    models_ready = health.get("models_loaded", False)
    if not models_ready:
        st.warning(
            "The API is running but no model artifacts are loaded. "
            "Train the models first: `make train` or `docker compose --profile train up`.",
            icon="⚠️",
        )
except APIError as exc:
    st.error(f"Prediction API unavailable: {exc}")
    st.info(
        "Start the stack with `docker compose up`, or run the API locally with "
        "`uvicorn wine_quality.serving.api:app --port 8000`."
    )

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
try:
    frame, profile = load_and_profile(settings)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.subheader("Dataset")

left, middle, right, far_right = st.columns(4)
left.metric("Samples", f"{profile.n_rows:,}")
middle.metric("Features", profile.n_features)
right.metric(
    f"'{'Good'}' wines (≥{profile.good_cutoff})",
    f"{profile.n_good_wines:,}",
    delta=f"{profile.positive_rate:.1%} of rows",
    delta_color="off",
)
far_right.metric("Duplicate rows", profile.n_duplicate_rows, delta_color="off")

st.caption(
    "The two tasks the dataset supports: **regression** on the 0–10 sensory score, "
    "and **binary classification** into good/bad using the cutoff above. "
    "Classes are *ordered and unbalanced* — the original authors flag this explicitly."
)

chart_left, chart_right = st.columns(2)

with chart_left:
    distribution = (
        frame[TARGET_NAME].value_counts().sort_index().rename_axis("quality").reset_index(name="count")
    )
    distribution["is_good"] = distribution["quality"] >= profile.good_cutoff
    figure = px.bar(
        distribution,
        x="quality",
        y="count",
        color="is_good",
        color_discrete_map={True: "#722F37", False: "#C9A9A6"},
        labels={"quality": "Quality score", "count": "Wines", "is_good": "Good wine"},
        title="Quality score distribution",
    )
    figure.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=0),
    )
    st.plotly_chart(figure, width="stretch")

with chart_right:
    balance = pd.DataFrame(
        {
            "Class": ["Not good", "Good"],
            "Count": [profile.n_rows - profile.n_good_wines, profile.n_good_wines],
        }
    )
    figure = px.pie(
        balance,
        names="Class",
        values="Count",
        hole=0.55,
        color="Class",
        color_discrete_map={"Not good": "#C9A9A6", "Good": "#722F37"},
        title="Binary class balance",
    )
    figure.update_traces(textinfo="percent+label")
    figure.update_layout(margin=dict(t=60, b=0), showlegend=False)
    st.plotly_chart(figure, width="stretch")

st.info(
    f"**Imbalance is the central modelling constraint.** Only {profile.positive_rate:.1%} of wines "
    f"clear the cutoff — a ratio of {profile.imbalance_ratio:.1f}:1. A model that predicts "
    f"'never good' scores ~{1 - profile.positive_rate:.1%} accuracy while finding zero good wines, "
    "which is why the classifier is ranked by **PR-AUC** rather than accuracy or ROC-AUC alone. "
    "Training uses `class_weight='balanced'` to compensate.",
    icon="⚖️",
)

# ---------------------------------------------------------------------------
# Served model
# ---------------------------------------------------------------------------
st.subheader("Served model")

if models_ready:
    try:
        info = get_model_info()
    except APIError as exc:
        st.error(f"Could not read model info: {exc}")
        st.stop()

    meta_left, meta_right = st.columns([1, 2])
    with meta_left:
        st.markdown(
            f"""
**Version** `{info['model_version']}`
**Trained** {info['trained_at']}
**Cutoff** quality ≥ {info['good_quality_cutoff']}
**Classifier** `{info['classifier_algorithm']}`
**Regressor** `{info['regressor_algorithm']}`
**Scikit-learn** `{info['sklearn_version']}`
"""
        )
        st.caption(
            f"Trained on {info['n_training_rows']:,} rows, "
            f"evaluated on {info['n_test_rows']:,} held-out rows."
        )

    with meta_right:
        clf = info["classification_metrics"]
        reg = info["regression_metrics"]
        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "ROC-AUC",
            f"{clf['roc_auc']:.3f}",
            help="Ranking quality. The Kaggle data card suggests ~0.88 is achievable without tuning.",
        )
        metric_columns[1].metric(
            "PR-AUC ★",
            f"{clf['average_precision']:.3f}",
            help="Average precision — the primary metric, because it reflects the minority class.",
        )
        metric_columns[2].metric("F1", f"{clf['f1']:.3f}", help="Precision/recall balance at 0.5.")

        error_columns = st.columns(3)
        error_columns[0].metric("RMSE", f"{reg['rmse']:.3f}", help="In quality-score units.")
        error_columns[1].metric("MAE", f"{reg['mae']:.3f}", help="In quality-score units.")
        error_columns[2].metric("R²", f"{reg['r2']:.3f}", help="Variance explained by the regressor.")

        st.caption(
            f"Confusion matrix [[TN, FP], [FN, TP]] = `{clf['confusion_matrix']}` — "
            f"{clf['true_positives']} good wines caught, {clf['false_negatives']} missed."
        )

    # -- Feature importance: answers the dataset's stated inspiration -----
    st.subheader("What makes a wine 'good'?")
    importances = _load_importances()

    if importances:
        importance_frame = pd.DataFrame(
            {"feature": list(importances.keys()), "importance": list(importances.values())}
        ).sort_values("importance")
        figure = go.Figure(
            go.Bar(
                x=importance_frame["importance"],
                y=importance_frame["feature"],
                orientation="h",
                marker_color="#722F37",
            )
        )
        figure.update_layout(
            title="Feature contribution to the 'good wine' verdict",
            xaxis_title="Relative importance",
            margin=dict(t=60, b=0, l=0),
            height=420,
        )
        st.plotly_chart(figure, width="stretch")
        top = importance_frame.iloc[-1]["feature"]
        st.caption(
            f"`{top}` is the single strongest signal. Note this reflects correlations in the "
            "training data, not causal effects — the dataset card explicitly warns against "
            "reading causation into these physicochemical measures."
        )
    else:
        st.caption(
            "Feature importances were not available. They are written to "
            "`reports/metrics.json` by the training run."
        )
else:
    st.caption(
        "Start the API and train the models to see live metrics here. "
        "The training run also writes metrics to `reports/metrics.json`."
    )

st.divider()
st.caption(
    "Navigate with the sidebar: **Explore the Data** for EDA, "
    "**Model Performance** for evaluation detail, **Predict** to score a wine."
)