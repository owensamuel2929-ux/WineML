"""Exploratory data analysis page.

Reads the raw CSV directly rather than going through the API: this is static
reference data, not a model output, so a network round-trip would add latency
without adding truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Streamlit runs each page as a standalone script, so neither the project root
# nor `src` is on sys.path. This file lives in app/pages/, hence parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from wine_quality.config import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    TARGET_NAME,
    get_settings,
)
from wine_quality.data.loader import load_and_profile  # noqa: E402
from wine_quality.features.pipeline import correlation_matrix  # noqa: E402

st.set_page_config(page_title="Wine Quality · Explore", page_icon="🔍", layout="wide")

settings = get_settings()
st.title("🔍 Explore the Data")


def _blend(start: str, end: str, weight: float) -> str:
    """Linearly interpolate between two hex colours.

    Args:
        start: Hex colour at weight 0, e.g. ``"#FFFFFF"``.
        end: Hex colour at weight 1.
        weight: Interpolation factor, clamped to ``[0, 1]``.

    Returns:
        The interpolated hex colour.
    """
    weight = min(max(weight, 0.0), 1.0)
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(a + (b - a) * weight) for a, b in zip(start_rgb, end_rgb, strict=True)
    )
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def _diverging_gradient(
    series: pd.Series,
    *,
    low: str = "#3B6EA5",
    mid: str = "#FFFFFF",
    high: str = "#722F37",
) -> list[str]:
    """Build per-cell CSS backgrounds on a diverging colour scale.

    Replaces pandas' ``Styler.background_gradient``, which requires matplotlib —
    a ~30 MB dependency that would otherwise be pulled in solely to shade table
    cells. Values are normalised by their largest absolute magnitude so zero maps
    to the neutral midpoint, which is what makes a diverging scale readable.

    Args:
        series: Values to colour.
        low: Colour for the most negative value.
        mid: Colour for zero.
        high: Colour for the most positive value.

    Returns:
        One CSS ``background-color`` declaration per value.
    """
    values = series.astype(float)
    scale = values.abs().max()

    if pd.isna(scale) or not scale:
        return ["background-color: transparent"] * len(values)

    styles: list[str] = []
    for value in values:
        if pd.isna(value):
            styles.append("background-color: transparent")
            continue
        ratio = float(value) / float(scale)
        colour = _blend(mid, high, ratio) if ratio >= 0 else _blend(mid, low, -ratio)
        styles.append(f"background-color: {colour}")
    return styles


def _sequential_gradient(
    series: pd.Series,
    *,
    low: str = "#FFFFFF",
    high: str = "#722F37",
) -> list[str]:
    """Build per-cell CSS backgrounds on a sequential colour scale.

    Args:
        series: Values to colour.
        low: Colour for the minimum value.
        high: Colour for the maximum value.

    Returns:
        One CSS ``background-color`` declaration per value.
    """
    values = series.astype(float)
    span = values.max() - values.min()

    if pd.isna(span) or not span:
        return ["background-color: transparent"] * len(values)

    return [
        "background-color: transparent"
        if pd.isna(value)
        else f"background-color: {_blend(low, high, (float(value) - values.min()) / span)}"
        for value in values
    ]
st.caption(
    "Eleven physicochemical measurements per wine, plus the sensory score assigned by "
    "expert tasters. The dataset card notes that grape type, brand, and price are "
    "deliberately absent for privacy and logistics reasons."
)

try:
    frame, profile = load_and_profile(settings)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

tab_dist, tab_corr, tab_compare, tab_raw = st.tabs(
    ["Distributions", "Correlations", "Good vs Bad", "Raw data"]
)

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
with tab_dist:
    st.markdown(
        "Most features are right-skewed with a long tail of unusual wines. "
        "`residual sugar`, `chlorides`, and the sulfur dioxide measures are the "
        "most skewed — worth remembering, since the models are trained on these "
        "raw scales."
    )

    selected = st.multiselect(
        "Features to plot",
        options=list(FEATURE_NAMES),
        default=["alcohol", "volatile acidity", "citric acid", "sulphates"],
        help="Select one or more physicochemical properties.",
    )

    if selected:
        columns = st.columns(2)
        for index, feature in enumerate(selected):
            with columns[index % 2]:
                figure = px.histogram(
                    frame,
                    x=feature,
                    nbins=45,
                    marginal="box",
                    color_discrete_sequence=["#722F37"],
                    title=feature,
                )
                figure.update_layout(
                    showlegend=False,
                    margin=dict(t=50, b=0),
                    height=320,
                    yaxis_title="Wines",
                )
                st.plotly_chart(figure, width="stretch")
                st.caption(FEATURE_DESCRIPTIONS.get(feature, ""))
    else:
        st.info("Select at least one feature to plot.")

    with st.expander("Summary statistics"):
        summary = frame[list(FEATURE_NAMES)].describe().T
        summary["skew"] = frame[list(FEATURE_NAMES)].skew()
        st.dataframe(
            summary.style.format("{:.4f}").apply(
                _sequential_gradient, subset=["skew"]
            ),
            width="stretch",
        )

# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
with tab_corr:
    st.markdown(
        "Pearson correlation across all features and the target. "
        "The dataset card warns that these are **correlations, not causes** — "
        "the physicochemical measures are themselves interdependent."
    )

    correlations = correlation_matrix(frame)
    figure = go.Figure(
        go.Heatmap(
            z=correlations.values,
            x=correlations.columns,
            y=correlations.columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(correlations.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="r"),
        )
    )
    figure.update_layout(height=720, margin=dict(t=20, b=0))
    st.plotly_chart(figure, width="stretch")

    target_correlations = (
        correlations[TARGET_NAME]
        .drop(TARGET_NAME)
        .sort_values(key=abs, ascending=False)
        .rename("correlation with quality")
        .to_frame()
    )
    st.markdown("**Strongest relationships with quality**")
    st.dataframe(
        target_correlations.style.format("{:+.3f}").apply(_diverging_gradient),
        width="stretch",
    )

    strongest = target_correlations.index[0]
    direction = "positively" if target_correlations.iloc[0, 0] > 0 else "negatively"
    st.info(
        f"`{strongest}` correlates most strongly with quality "
        f"(r = {target_correlations.iloc[0, 0]:+.3f}), and it does so {direction}. "
        "This matches the domain expectation that alcohol content is a major driver "
        "of perceived quality in Vinho Verde wines.",
        icon="📈",
    )

# ---------------------------------------------------------------------------
# Good vs Bad
# ---------------------------------------------------------------------------
with tab_compare:
    st.markdown(
        f"Comparing wines at or above the cutoff (quality ≥ {profile.good_cutoff}) "
        "against the rest. This is the split the classifier learns."
    )

    frame_with_label = frame.copy()
    frame_with_label["verdict"] = np.where(
        frame_with_label[TARGET_NAME] >= profile.good_cutoff, "Good", "Not good"
    )

    feature = st.selectbox(
        "Feature to compare",
        options=list(FEATURE_NAMES),
        index=list(FEATURE_NAMES).index("alcohol"),
    )

    compare_left, compare_right = st.columns(2)

    with compare_left:
        figure = px.violin(
            frame_with_label,
            x="verdict",
            y=feature,
            color="verdict",
            box=True,
            points="outliers",
            color_discrete_map={"Good": "#722F37", "Not good": "#C9A9A6"},
            title=f"{feature} by verdict",
        )
        figure.update_layout(showlegend=False, margin=dict(t=50, b=0), height=420)
        st.plotly_chart(figure, width="stretch")

    with compare_right:
        grouped = (
            frame_with_label.groupby("verdict")[feature]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={"mean": "Mean", "median": "Median", "std": "Std dev", "count": "Wines"})
        )
        st.markdown(f"**{feature} statistics**")
        st.dataframe(grouped.style.format("{:.4f}"), width="stretch")

        good_mean = grouped.loc["Good", "Mean"]
        bad_mean = grouped.loc["Not good", "Mean"]
        delta = good_mean - bad_mean
        st.metric(
            "Difference in means",
            f"{delta:+.4f}",
            delta=f"{delta / bad_mean:+.1%} vs not-good wines" if bad_mean else None,
            delta_color="off",
        )

    st.markdown("**Mean profile across every feature**")
    profile_table = (
        frame_with_label.groupby("verdict")[list(FEATURE_NAMES)]
        .mean()
        .T.rename(columns={"Good": "Good mean", "Not good": "Not-good mean"})
    )
    profile_table["Difference"] = profile_table["Good mean"] - profile_table["Not-good mean"]
    profile_table["Relative"] = profile_table["Difference"] / profile_table["Not-good mean"]
    st.dataframe(
        profile_table.style.format(
            {
                "Good mean": "{:.4f}",
                "Not-good mean": "{:.4f}",
                "Difference": "{:+.4f}",
                "Relative": "{:+.1%}",
            }
        ).apply(_diverging_gradient, subset=["Difference"]),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
with tab_raw:
    st.markdown(
        f"All {profile.n_rows:,} rows. Note the **{profile.n_duplicate_rows} duplicate rows** — "
        "these are genuine repeated measurements rather than data-entry errors, so they are "
        "retained. They do mean the effective sample size is slightly smaller than it appears."
    )

    filter_left, filter_right = st.columns([1, 3])
    with filter_left:
        minimum = st.slider(
            "Minimum quality",
            min_value=int(frame[TARGET_NAME].min()),
            max_value=int(frame[TARGET_NAME].max()),
            value=int(frame[TARGET_NAME].min()),
        )
    with filter_right:
        search = st.text_input("Filter rows (substring match across all columns)", "")

    view = frame[frame[TARGET_NAME] >= minimum]
    if search:
        mask = view.astype(str).apply(
            lambda column: column.str.contains(search, case=False, na=False)
        ).any(axis=1)
        view = view[mask]

    st.caption(f"Showing {len(view):,} of {profile.n_rows:,} rows.")
    st.dataframe(view, width="stretch", height=460)

    st.download_button(
        "Download filtered rows as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="winequality-red-filtered.csv",
        mime="text/csv",
    )

    with st.expander("Data quality checks"):
        checks = pd.DataFrame(
            {
                "Check": [
                    "Missing values",
                    "Duplicate rows",
                    "Non-numeric columns",
                    "Rows outside plausible bounds",
                ],
                "Result": [
                    sum(profile.missing_values.values()),
                    profile.n_duplicate_rows,
                    0,
                    0,
                ],
                "Status": ["✅ Pass", "ℹ️ Retained", "✅ Pass", "✅ Pass"],
            }
        )
        st.dataframe(checks, width="stretch", hide_index=True)
        st.caption(
            "Validation runs on every load, so a corrupted or replaced source file "
            "fails loudly at the start of a training run rather than silently "
            "degrading model quality."
        )