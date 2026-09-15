"""Model evaluation page.

Reads the artifacts written by the training run: held-out metrics, the
confusion matrix, cross-validation scores per candidate, and the ROC and
precision-recall curves. Everything here is a *held-out* result — no training-set
numbers are shown, since they would flatter the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

# Streamlit runs each page as a standalone script, so neither the project root
# nor `src` is on sys.path. This file lives in app/pages/, hence parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from app.api_client import APIError, get_model_info  # noqa: E402

from wine_quality.config import get_settings  # noqa: E402
from wine_quality.data.loader import load_and_profile  # noqa: E402
from wine_quality.data.schema import build_targets  # noqa: E402
from wine_quality.models.registry import ArtifactError, load_artifacts  # noqa: E402

st.set_page_config(page_title="Wine Quality · Performance", page_icon="📊", layout="wide")

settings = get_settings()
st.title("📊 Model Performance")
st.caption(
    "All figures below are computed on the held-out test split, which the models "
    "never saw during training or model selection."
)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
try:
    classifier, regressor, metadata = load_artifacts(settings)
except ArtifactError as exc:
    st.error(str(exc))
    st.info("Run `make train` to produce the artifacts, then reload this page.")
    st.stop()

try:
    frame, profile = load_and_profile(settings)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

# Rebuild the identical split so curves can be drawn. The seed and test size come
# from settings, so this reproduces the training split exactly.
features, quality, is_good = build_targets(frame, settings)
_, x_test, _, y_quality_test, _, y_good_test = train_test_split(
    features,
    quality,
    is_good,
    test_size=settings.test_size,
    random_state=settings.random_state,
    stratify=is_good,
)

y_proba = classifier.predict_proba(x_test)[:, 1]
y_pred = classifier.predict(x_test)
y_quality_pred = regressor.predict(x_test)

clf_metrics = metadata.classification_metrics
reg_metrics = metadata.regression_metrics

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
st.subheader("Headline metrics")

st.markdown("**Classification — is this a good wine?**")
clf_columns = st.columns(4)
clf_columns[0].metric(
    "PR-AUC ★",
    f"{clf_metrics['average_precision']:.3f}",
    help="Average precision. Primary metric: robust to the 13.6% positive rate.",
)
clf_columns[1].metric(
    "ROC-AUC",
    f"{clf_metrics['roc_auc']:.3f}",
    help="Ranking quality across all thresholds.",
)
clf_columns[2].metric("F1", f"{clf_metrics['f1']:.3f}", help="At the 0.5 threshold.")
clf_columns[3].metric("Accuracy", f"{clf_metrics['accuracy']:.3f}", help="Misleading here — see below.")

st.markdown("**Regression — what score will it get?**")
reg_columns = st.columns(4)
reg_columns[0].metric("RMSE", f"{reg_metrics['rmse']:.3f}", help="In quality-score units.")
reg_columns[1].metric("MAE", f"{reg_metrics['mae']:.3f}", help="In quality-score units.")
reg_columns[2].metric("R²", f"{reg_metrics['r2']:.3f}", help="Variance explained.")
reg_columns[3].metric("Max error", f"{reg_metrics['max_error']:.3f}", help="Worst single prediction.")

baseline = 1 - profile.positive_rate
st.warning(
    f"**Read accuracy with suspicion.** A trivial model that labels every wine "
    f"'not good' would score {baseline:.1%} accuracy while catching zero good wines. "
    f"The selected model reaches {clf_metrics['accuracy']:.1%} accuracy *and* "
    f"{clf_metrics['recall']:.1%} recall on the minority class — that combination is "
    "what makes it useful. This is why PR-AUC is the primary metric.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------
st.subheader("Threshold behaviour")

curve_left, curve_right = st.columns(2)

with curve_left:
    fpr, tpr, _ = roc_curve(y_good_test, y_proba)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color="#722F37", width=3))
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random (AUC 0.5)",
            line=dict(color="#999", dash="dash"),
        )
    )
    figure.update_layout(
        title=f"ROC curve (AUC = {clf_metrics['roc_auc']:.3f})",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        margin=dict(t=60, b=0),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(figure, width="stretch")

with curve_right:
    precision, recall, _ = precision_recall_curve(y_good_test, y_proba)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recall, y=precision, mode="lines", name="Model", line=dict(color="#722F37", width=3)
        )
    )
    figure.add_hline(
        y=profile.positive_rate,
        line_dash="dash",
        line_color="#999",
        annotation_text=f"No-skill baseline ({profile.positive_rate:.3f})",
        annotation_position="bottom right",
    )
    figure.update_layout(
        title=f"Precision-recall curve (AP = {clf_metrics['average_precision']:.3f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(t=60, b=0),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(figure, width="stretch")

st.caption(
    "The PR curve is the more informative of the two here. Its no-skill baseline sits at the "
    f"positive rate ({profile.positive_rate:.3f}) rather than 0.5, so the gap between the curve "
    "and that line is the model's real contribution."
)

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
st.subheader("Confusion matrix")

matrix_left, matrix_right = st.columns([1, 1])

with matrix_left:
    cm = np.array(clf_metrics["confusion_matrix"])
    figure = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted not good", "Predicted good"],
            y=["Actually not good", "Actually good"],
            colorscale="Reds",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=False,
        )
    )
    figure.update_layout(height=380, margin=dict(t=20, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(figure, width="stretch")

with matrix_right:
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    st.markdown(
        f"""
| Outcome | Count | Meaning |
|---|---:|---|
| True positives | {tp} | Good wines correctly flagged |
| False negatives | {fn} | Good wines the model missed |
| False positives | {fp} | Ordinary wines flagged as good |
| True negatives | {tn} | Ordinary wines correctly passed over |
"""
    )
    st.caption(
        f"Of {tp + fn} genuinely good wines in the test set, the model found {tp} "
        f"({clf_metrics['recall']:.1%}). Of the {tp + fp} wines it flagged, {tp} were "
        f"genuinely good ({clf_metrics['precision']:.1%})."
    )
    st.caption(
        "Which error matters more is a business decision: a false positive wastes a "
        "premium listing on an ordinary wine, while a false negative buries a good one."
    )

# ---------------------------------------------------------------------------
# Regression diagnostics
# ---------------------------------------------------------------------------
st.subheader("Regression diagnostics")

reg_left, reg_right = st.columns(2)

with reg_left:
    figure = px.scatter(
        x=y_quality_test,
        y=y_quality_pred,
        opacity=0.45,
        labels={"x": "Actual quality", "y": "Predicted quality"},
        color_discrete_sequence=["#722F37"],
        title="Predicted vs actual",
    )
    limits = [y_quality_test.min() - 0.5, y_quality_test.max() + 0.5]
    figure.add_trace(
        go.Scatter(
            x=limits,
            y=limits,
            mode="lines",
            name="Perfect prediction",
            line=dict(color="#999", dash="dash"),
        )
    )
    figure.update_layout(margin=dict(t=60, b=0), height=420, showlegend=False)
    st.plotly_chart(figure, width="stretch")

with reg_right:
    residuals = np.asarray(y_quality_test) - np.asarray(y_quality_pred)
    figure = px.histogram(
        x=residuals,
        nbins=40,
        labels={"x": "Residual (actual − predicted)"},
        color_discrete_sequence=["#722F37"],
        title="Residual distribution",
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#999")
    figure.update_layout(margin=dict(t=60, b=0), height=420, yaxis_title="Count")
    st.plotly_chart(figure, width="stretch")

st.caption(
    f"Residuals are centred near zero (mean {reg_metrics['mean_residual']:+.3f}) with a spread of "
    f"{reg_metrics['std_residual']:.3f}. The model regresses toward the mean — it rarely predicts "
    "the extremes (3 or 8), which is expected given how few such wines exist. "
    f"R² of {reg_metrics['r2']:.3f} means most score variation is driven by factors this dataset "
    "does not capture, such as grape variety and winemaking technique."
)

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
st.subheader("Model selection")

metrics_path = settings.metrics_path
if metrics_path.exists():
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    cv_scores = payload.get("cv_scores", {})
    selected = payload.get("selected_models", {})

    selection_left, selection_right = st.columns(2)

    with selection_left:
        if "classifier" in cv_scores:
            cv_frame = (
                pd.DataFrame(
                    {
                        "Candidate": list(cv_scores["classifier"].keys()),
                        "CV PR-AUC": list(cv_scores["classifier"].values()),
                    }
                )
                .sort_values("CV PR-AUC", ascending=False)
                .reset_index(drop=True)
            )
            cv_frame["Selected"] = cv_frame["Candidate"] == selected.get("classifier")
            figure = px.bar(
                cv_frame,
                x="CV PR-AUC",
                y="Candidate",
                orientation="h",
                color="Selected",
                color_discrete_map={True: "#722F37", False: "#C9A9A6"},
                title="Classifier candidates (5-fold CV)",
            )
            figure.update_layout(
                margin=dict(t=60, b=0), height=320, showlegend=False, yaxis_title=""
            )
            st.plotly_chart(figure, width="stretch")

    with selection_right:
        if "regressor" in cv_scores:
            cv_frame = (
                pd.DataFrame(
                    {
                        "Candidate": list(cv_scores["regressor"].keys()),
                        "CV RMSE": [-value for value in cv_scores["regressor"].values()],
                    }
                )
                .sort_values("CV RMSE")
                .reset_index(drop=True)
            )
            cv_frame["Selected"] = cv_frame["Candidate"] == selected.get("regressor")
            figure = px.bar(
                cv_frame,
                x="CV RMSE",
                y="Candidate",
                orientation="h",
                color="Selected",
                color_discrete_map={True: "#722F37", False: "#C9A9A6"},
                title="Regressor candidates (5-fold CV, lower is better)",
            )
            figure.update_layout(
                margin=dict(t=60, b=0), height=320, showlegend=False, yaxis_title=""
            )
            st.plotly_chart(figure, width="stretch")

    st.caption(
        f"Selected: **{selected.get('classifier', 'n/a')}** for classification and "
        f"**{selected.get('regressor', 'n/a')}** for regression. Candidates are ranked by "
        "cross-validated score on the training split only; the test split is touched once, "
        "after selection, so the reported metrics are not optimistically biased."
    )
else:
    st.info("No `reports/metrics.json` found. Run training to populate model-selection results.")

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
with st.expander("Run provenance"):
    st.markdown(
        f"""
| Field | Value |
|---|---|
| Model version | `{metadata.model_version}` |
| Trained at | {metadata.trained_at} |
| Good-wine cutoff | quality ≥ {metadata.good_quality_cutoff} |
| Training rows | {metadata.n_training_rows:,} |
| Test rows | {metadata.n_test_rows:,} |
| Classifier | `{metadata.classifier_algorithm}` |
| Regressor | `{metadata.regressor_algorithm}` |
| Scikit-learn | `{metadata.sklearn_version}` |
| Python | `{metadata.python_version}` |
"""
    )
    st.caption(
        "Feature order is pinned in the artifact metadata. If the schema ever changes, "
        "the mismatch is detectable rather than silent."
    )

    try:
        live = get_model_info()
        if live["trained_at"] != metadata.trained_at:
            st.warning(
                "The API is serving a **different** model than the artifacts on disk. "
                "Call `POST /reload` to pick up the newer model.",
                icon="🔄",
            )
        else:
            st.success("The API is serving the same model as the artifacts on disk.", icon="✅")
    except APIError:
        st.caption("API not reachable — cannot compare the served model against disk artifacts.")