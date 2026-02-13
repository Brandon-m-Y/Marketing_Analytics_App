# pages/2_🔎_EDA.py
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load environment variables
load_dotenv()
# LLM_API_KEY = os.getenv("LLM_API_KEY")

st.set_page_config(
    page_title= "Exploratory Data Analysis (EDA)",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Exploratory Data Analysis (EDA)")
st.write("This page explores the dataset loaded on **📥 Data Intake**.")

# -----------------------------
# Require data from Page A
# -----------------------------

fe_df = st.session_state.get("fe_df")
clean_df = st.session_state.get("clean_df")
raw_df = st.session_state.get("raw_df")

if fe_df is not None:
    df = fe_df
    source_label = "Feature Engineered Data"
elif clean_df is not None:
    df = clean_df
    source_label = "Cleaned Data"
elif raw_df is not None:
    df = raw_df
    source_label = "Raw Data"
else:
    df = None


if df is None:
    st.warning("No dataset found. Please upload one in **📥 Data Intake** first.")
    if st.button("➡️ Go to Data Intake"):
        st.switch_page("pages/1_📥_Data_Intake.py")
    st.stop()

st.caption(f"File: {st.session_state.get('file_name', 'Unknown')}")
st.write("DEBUG columns:", df.columns.tolist())

# =========================================================
# Explain Graph: UI + (future) LLM hook
# =========================================================
def explain_graph_ui(
    section_key: str,
    fig=None,
    context: dict | None = None,
    env_var_name: str = "LLM_API_KEY",
):
    """
    Render a per-chart "Explain this graph" UI block.

    Current behavior:
    - Shows a button labeled: 'Explain this graph to me (Not Yet available)'
    - Disabled unless an API key is found in environment variables AND fig is provided
      (so you're set up to plug in an LLM later without changing UI layout).

    Future behavior (you'll implement in `explain_with_llm`):
    - Serialize Plotly figure to JSON + attach context + send to an LLM endpoint
    - Store/display the response below the chart
    """
    # Where the explanation will be stored and retrieved
    resp_key = f"explain_response_{section_key}"

    # "Ready" check: you can change env_var_name to match your setup later.
    api_key_present = bool(os.getenv(env_var_name))
    can_run = api_key_present and (fig is not None)

    # Button label matches your requirement exactly
    st.button(
        "Explain this graph to me (Not Yet available)",
        key=f"btn_explain_{section_key}",
        disabled=not can_run,  # stays disabled until API key exists + fig provided
        help=(
            "Coming soon: sends this chart + dataset context to an LLM and returns an explanation."
            if not can_run
            else "API key detected. Wire up the LLM call in `explain_with_llm()`."
        ),
    )

    # If you want the UI to show a placeholder response area consistently:
    if resp_key in st.session_state and st.session_state[resp_key]:
        st.markdown("**Explanation**")
        st.write(st.session_state[resp_key])
    else:
        st.caption("Explanation output will appear here once the feature is enabled.")

    # ---- FUTURE LLM CALL (stub) ----
    # When you're ready:
    # 1) Change the button text to remove '(Not Yet available)'
    # 2) Enable the button unconditionally
    # 3) Implement `explain_with_llm(fig, context, api_key)`
    #
    # Example skeleton:
    #
    # if st.button("Explain this graph to me", key=f"btn_explain_{section_key}"):
    #     api_key = os.getenv(env_var_name)
    #     response = explain_with_llm(fig, context or {}, api_key)
    #     st.session_state[resp_key] = response
    #     st.rerun()


# -----------------------------
# Cached summary helpers
# -----------------------------
#@st.cache_data
def basic_overview(_df: pd.DataFrame) -> dict:
    return {
        "rows": int(_df.shape[0]),
        "cols": int(_df.shape[1]),
        "missing_cells": int(_df.isna().sum().sum()),
        "duplicate_rows": int(_df.duplicated().sum()),
        "memory_mb": float(_df.memory_usage(deep=True).sum() / (1024**2)),
    }

#@st.cache_data
def missingness_table(_df: pd.DataFrame) -> pd.DataFrame:
    miss = _df.isna().sum()
    pct = (miss / len(_df)).replace([np.inf], np.nan) * 100
    return (
        pd.DataFrame({"missing_count": miss, "missing_pct": pct})
        .sort_values("missing_pct", ascending=False)
    )

#@st.cache_data
def split_columns(_df: pd.DataFrame) -> dict:
    numeric_cols = _df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in _df.columns if c not in numeric_cols]
    return {"numeric": numeric_cols, "non_numeric": non_numeric_cols}

#@st.cache_data
def corr_matrix(_df: pd.DataFrame) -> pd.DataFrame:
    num = _df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(numeric_only=True)


# -----------------------------
# A) Overview
# -----------------------------
st.subheader("A) Overview")

ov = basic_overview(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{ov['rows']:,}")
c2.metric("Columns", f"{ov['cols']:,}")
c3.metric("Missing cells", f"{ov['missing_cells']:,}")
c4.metric("Duplicate rows", f"{ov['duplicate_rows']:,}")
c5.metric("Memory (MB)", f"{ov['memory_mb']:.2f}")

with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)

cols = split_columns(df)
numeric_cols = cols["numeric"]
non_numeric_cols = cols["non_numeric"]


# -----------------------------
# B) Missingness
# -----------------------------
st.subheader("B) Missingness")

miss_tbl = missingness_table(df)
st.dataframe(miss_tbl, use_container_width=True)

top_n = st.slider("Show top N columns by missing %", 5, min(50, len(df.columns)), 15)
miss_top = miss_tbl.head(top_n).reset_index().rename(columns={"index": "column"})

fig_miss = px.bar(
    miss_top,
    x="column",
    y="missing_pct",
    hover_data=["missing_count"],
    title=f"Top {top_n} columns by missing %",
)
fig_miss.update_layout(xaxis_tickangle=-60)
st.plotly_chart(fig_miss, use_container_width=True)

# Explain button under missingness chart
explain_graph_ui(
    section_key="missingness_bar",
    fig=fig_miss,
    context={
        "chart_type": "missingness_bar",
        "top_n": top_n,
        "notes": "Shows percent missing per column for the top N columns."
    }
)


# -----------------------------
# C) Histogram explorer
# -----------------------------
st.subheader("C) Histogram Explorer")

if len(numeric_cols) == 0:
    st.info("No numeric columns detected. Histograms require numeric columns.")
else:
    st.write("Choose a numeric feature below to generate a histogram:")

    hist_col = st.selectbox("Histogram column", numeric_cols)
    bins = st.slider("Bins", 5, 100, 30)

    series = df[hist_col].dropna()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Count", f"{series.shape[0]:,}")
    s2.metric("Mean", f"{series.mean():,.1f}" if series.shape[0] else "—")
    s3.metric("Std", f"{series.std():,.1f}" if series.shape[0] else "—")
    s4.metric("Unique", f"{series.nunique():,}")

    fig_hist = px.histogram(
        df,
        x=hist_col,
        nbins=bins,
        title=f"Distribution of {hist_col}",
        
    )
    fig_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Explain button under histogram
    explain_graph_ui(
        section_key="histogram",
        fig=fig_hist,
        context={
            "chart_type": "histogram",
            "column": hist_col,
            "bins": bins,
            "summary_stats": {
                "count": int(series.shape[0]),
                "mean": float(series.mean()) if series.shape[0] else None,
                "std": float(series.std()) if series.shape[0] else None,
                "min": float(series.min()) if series.shape[0] else None,
                "max": float(series.max()) if series.shape[0] else None,
            },
            "notes": "Distribution of the selected numeric feature."
        }
    )

    fig_box = px.box(df, x=hist_col, title=f"Boxplot: {hist_col}")
    st.plotly_chart(fig_box, use_container_width=True)

    # Explain button under boxplot
    explain_graph_ui(
        section_key="boxplot",
        fig=fig_box,
        context={
            "chart_type": "boxplot",
            "column": hist_col,
            "notes": "Boxplot highlights median, IQR, and potential outliers."
        }
    )


# -----------------------------
# D) Scatter plot explorer
# -----------------------------
st.subheader("D) Scatter Plot Explorer")

if len(numeric_cols) < 2:
    st.info("Need at least 2 numeric columns to create a scatter plot.")
else:
    st.write("Choose X and Y features to explore relationships:")

    left, right = st.columns(2)
    with left:
        x_col = st.selectbox("X axis", numeric_cols, index=0)
    with right:
        y_default = 1 if len(numeric_cols) > 1 else 0
        y_col = st.selectbox("Y axis", numeric_cols, index=y_default)

    color_options = ["(none)"] + df.columns.tolist()
    color_col = st.selectbox("Color by (optional)", color_options, index=0)

    used_cols = [x_col, y_col] + ([] if color_col == "(none)" else [color_col])
    plot_df = df[used_cols].dropna()

    if plot_df.empty:
        st.warning("No rows left after dropping missing values for the selected columns.")
    else:
        if color_col == "(none)":
            fig_scatter = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}",
            )
        else:
            fig_scatter = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col} (colored by {color_col})",
            )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # Explain button under scatter
        explain_graph_ui(
            section_key="scatter",
            fig=fig_scatter,
            context={
                "chart_type": "scatter",
                "x": x_col,
                "y": y_col,
                "color_by": None if color_col == "(none)" else color_col,
                "rows_plotted": int(plot_df.shape[0]),
                "notes": "Scatter plot to inspect relationship/trend between two numeric variables."
            }
        )


# -----------------------------
# E) Correlations (numeric only)
# -----------------------------
st.subheader("E) Correlations (numeric only)")

corr = corr_matrix(df)

if corr.empty:
    st.info("Need at least 2 numeric columns to compute correlations.")
else:
    fig_corr = px.imshow(
        corr,
        aspect="auto",
        title="Correlation matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Explain button under correlation heatmap
    explain_graph_ui(
        section_key="corr_heatmap",
        fig=fig_corr,
        context={
            "chart_type": "correlation_heatmap",
            "num_columns": corr.shape[0],
            "notes": "Correlation between numeric features (Pearson by default in pandas)."
        }
    )


# -----------------------------
# F) Export
# -----------------------------
st.subheader("F) Export")

csv_bytes = miss_tbl.to_csv(index=True).encode("utf-8")
st.download_button(
    "Download missingness table (CSV)",
    data=csv_bytes,
    file_name="missingness_table.csv",
    mime="text/csv"
)

if st.button("➡️ Continue to Customer Segmentation"):
        st.switch_page("pages/5_🧩_Customer_Segmentation.py")
