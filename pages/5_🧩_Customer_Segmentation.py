# pages/4_🧩_Customer_Segmentation.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# Optional elbow detection dependency
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except Exception:
    KNEED_AVAILABLE = False


# =========================================================
# Page header
# =========================================================
st.title("🧩 Customer Segmentation (K-Means)")
st.write("Group your data into distinct segments to identify patterns and opportunities.")


# =========================================================
# Data requirement: prefer fe_df, else clean_df, else raw_df
# =========================================================
fe_df = st.session_state.get("fe_df")
clean_df = st.session_state.get("clean_df")
raw_df = st.session_state.get("raw_df")

if fe_df is not None:
    df = fe_df
    df_source = "fe_df"
elif clean_df is not None:
    df = clean_df
    df_source = "clean_df"
elif raw_df is not None:
    df = raw_df
    df_source = "raw_df"
else:
    df = None
    df_source = "none"

if df is None:
    st.warning("No dataset found. Please upload data in **📥 Data Intake** and run **🧹 Data Cleaning** first.")
    if st.button("➡️ Go to Data Intake"):
        st.switch_page("pages/1_📥_Data_Intake.py")
    st.stop()

st.caption(f"Using: **{df_source}** | File: {st.session_state.get('file_name', 'Unknown')}")


# =========================================================
# LLM UI stubs (disabled placeholders)
# =========================================================
def explain_graph_ui(section_key: str, fig=None, context: dict | None = None, env_var_name: str = "LLM_API_KEY"):
    api_key_present = bool(os.getenv(env_var_name))
    can_run = api_key_present and (fig is not None)

    st.button(
        "🤖 Explain this chart to me (Not Yet available)",
        key=f"btn_explain_{section_key}",
        disabled=not can_run,
        help="Coming soon: sends chart + context to an LLM and returns a business-friendly explanation.",
    )
    st.caption("Explanation output will appear here once enabled.")


def ai_recommendations_ui(section_key: str, payload: dict | None = None, env_var_name: str = "LLM_API_KEY"):
    api_key_present = bool(os.getenv(env_var_name))
    can_run = api_key_present and (payload is not None)

    st.button(
        "🤖 Get AI recommendations for these segments (Not Yet available)",
        key=f"btn_ai_{section_key}",
        disabled=not can_run,
        help="Coming soon: sends segment summaries to an LLM for naming + strategies.",
    )
    st.caption("AI recommendations will appear here once enabled.")


# =========================================================
# 1) Introduction & Use Case Selection
# =========================================================
st.subheader("1) What are you trying to segment?")
use_case = st.selectbox(
    "Choose a segmentation use case",
    ["Customers", "Products", "Locations", "Sales Data", "Other"],
    index=0,
)

use_case_label = {
    "Customers": "customer segments",
    "Products": "product segments",
    "Locations": "location segments",
    "Sales Data": "sales segments",
    "Other": "segments",
}[use_case]

st.write(
    "K-Means builds segments using **multiple numeric features**. "
    "Optionally choose a **primary metric** to interpret and visualize the segments "
    "(e.g., “sales-level segments”)."
)


# =========================================================
# Helpers: smart feature filtering + transparency
# =========================================================
def looks_like_id_column(col_name: str) -> bool:
    name = col_name.strip().lower()
    return "id" in name or name.endswith("_id") or name.startswith("id_") or name == "id"


def is_sequential(series: pd.Series) -> bool:
    """
    Heuristic: detects simple row-number-like sequences.
    - numeric
    - mostly unique
    - values increase by ~1 after sorting
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna()
    if s.empty:
        return False
    if s.nunique() < max(10, int(0.8 * len(s))):
        return False
    vals = np.sort(s.values)
    diffs = np.diff(vals)
    return (np.mean(diffs == 1) >= 0.90) and (np.nanmin(diffs) >= 1)


def auto_select_numeric_features(df_in: pd.DataFrame):
    """
    Select numeric features and auto-filter:
    - ID-like columns that are all unique
    - zero-variance columns
    - sequential row-number columns
    Returns: (features_df, used_cols, filtered_info_list)
    """
    numeric_cols = df_in.select_dtypes(include=["number"]).columns.tolist()

    filtered = []
    used = []

    for col in numeric_cols:
        s = df_in[col]

        if s.dropna().empty:
            filtered.append((col, "All values missing"))
            continue

        if s.nunique(dropna=True) <= 1:
            filtered.append((col, "Zero variance (same value)"))
            continue

        if looks_like_id_column(col) and s.nunique(dropna=True) == s.dropna().shape[0]:
            filtered.append((col, "Looks like an ID (name contains 'id' and values are unique)"))
            continue

        if is_sequential(s):
            filtered.append((col, "Looks like a row-number / sequential index"))
            continue

        used.append(col)

    features_df = df_in[used].copy() if used else pd.DataFrame()
    filtered_info = [{"feature": f, "reason": r} for (f, r) in filtered]
    return features_df, used, filtered_info


# =========================================================
# 2) Automatic Feature Preparation
# =========================================================
st.subheader("2) Feature preparation (automatic)")

features_df_auto, used_features, filtered_info = auto_select_numeric_features(df)

if len(df.select_dtypes(include=["number"]).columns) < 2:
    st.error("Segmentation requires at least **2 numeric columns**. Please upload data with more numeric metrics.")
    st.stop()

if features_df_auto.empty or len(used_features) < 2:
    st.error("No suitable numeric features found after auto-filtering. Try manual feature selection in **Advanced Options**.")
    with st.expander("Auto-filtered features"):
        st.dataframe(pd.DataFrame(filtered_info), use_container_width=True)
    st.stop()

st.success(f"Using **{len(used_features)} features** for segmentation.")
with st.expander("View features used for segmentation"):
    st.write(used_features)

with st.expander("Auto-filtered features (and why)"):
    if filtered_info:
        st.dataframe(pd.DataFrame(filtered_info), use_container_width=True)
    else:
        st.caption("No features were auto-filtered.")


# =========================================================
# 9) Advanced Options (collapsed)
# =========================================================
with st.expander("Advanced Options (optional)", expanded=False):
    st.write("Override defaults if you want more control.")

    manual_override = st.checkbox("Manually choose features (override auto-selection)", value=False)

    if manual_override:
        all_numeric = df.select_dtypes(include=["number"]).columns.tolist()
        manual_features = st.multiselect("Select numeric features to use", all_numeric, default=used_features)
    else:
        manual_features = used_features

    k_min, k_max = st.slider("Elbow K range", min_value=2, max_value=25, value=(2, 10))
    random_state = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    n_init = st.selectbox("K-Means initializations (n_init)", options=[10, 20, 50], index=0)

    impute_strategy = st.selectbox(
        "Missing value handling (imputation strategy)",
        options=["median", "mean", "most_frequent"],
        index=0,
        help="K-Means can't accept missing values. We fill missing numeric values before scaling/clustering.",
    )

# Final features used for clustering
final_features = manual_features if manual_override else used_features
if final_features is None or len(final_features) < 2:
    st.error("Please select at least 2 numeric features.")
    st.stop()

features_df = df[final_features].copy()

# Primary metric selection
st.subheader("Primary metric (for interpretation)")
primary_metric = st.selectbox(
    "Choose a primary metric to interpret segments (used for summaries & charts)",
    options=final_features,
    index=0,
    help="K-Means uses all features to form segments. This metric helps you interpret and label what the segments mean.",
)


# =========================================================
# 3) Data preprocessing (impute + scale)
# =========================================================
st.subheader("3) Data preprocessing")

missing_cells = int(features_df.isna().sum().sum())
if missing_cells > 0:
    st.warning(
        f"Found **{missing_cells:,} missing values** in segmentation features. "
        f"They will be filled using **{impute_strategy} imputation** before clustering."
    )

st.info("✓ Features are imputed (if needed) and normalized so all features have equal weight.")

@st.cache_data(show_spinner=False)
def impute_and_scale(values: np.ndarray, strategy: str):
    imputer = SimpleImputer(strategy=strategy)
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(values)
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled

X = impute_and_scale(features_df.values, strategy=impute_strategy)


# =========================================================
# 4) Automatic Elbow Analysis
# =========================================================
st.subheader("4) Automatic elbow analysis")

@st.cache_data(show_spinner="Analyzing optimal number of segments...")
def elbow_analysis(X_in: np.ndarray, k_min_in: int, k_max_in: int, rs: int, n_init_in: int):
    inertias = []
    ks = list(range(k_min_in, k_max_in + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=rs, n_init=n_init_in)
        km.fit(X_in)
        inertias.append(float(km.inertia_))
    return ks, inertias

ks, inertias = elbow_analysis(X, k_min, k_max, int(random_state), int(n_init))

elbow_df = pd.DataFrame({"k": ks, "inertia": inertias})
fig_elbow = px.line(elbow_df, x="k", y="inertia", markers=True, title="Elbow method: inertia vs number of segments")

# Detect elbow point using kneed if available
recommended_k = None
knee_note = ""

if KNEED_AVAILABLE:
    try:
        kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
        recommended_k = int(kl.knee) if kl.knee is not None else None
        knee_note = "Detected with kneed" if recommended_k is not None else "kneed could not find a clear elbow"
    except Exception:
        recommended_k = None
        knee_note = "kneed elbow detection failed"
else:
    knee_note = "kneed not installed (elbow detection unavailable)"

if recommended_k is not None:
    st.success(f"📊 Recommended: **{recommended_k} segments** based on elbow analysis. ({knee_note})")
    star_df = elbow_df[elbow_df["k"] == recommended_k]
    if not star_df.empty:
        fig_star = px.scatter(star_df, x="k", y="inertia")
        fig_elbow.add_traces(fig_star.data)
        fig_elbow.data[-1].update(marker={"size": 14, "symbol": "star"})
    st.plotly_chart(fig_elbow, use_container_width=True)
else:
    st.warning(f"Could not automatically detect optimal clusters. ({knee_note})")
    st.info("You can choose K manually below.")
    st.plotly_chart(fig_elbow, use_container_width=True)

explain_graph_ui(
    section_key="elbow_chart",
    fig=fig_elbow,
    context={
        "use_case": use_case,
        "use_case_label": use_case_label,
        "features_used": final_features,
        "primary_metric": primary_metric,
        "k_range": [k_min, k_max],
        "inertias": inertias,
        "recommended_k": recommended_k,
        "imputation_strategy": impute_strategy,
        "missing_cells_in_features": missing_cells,
    },
)


# =========================================================
# 5) User Choice with Guidance
# =========================================================
st.subheader("5) Choose number of segments (guided)")

if recommended_k is None:
    chosen_k = st.slider("Choose number of segments (K)", min_value=k_min, max_value=k_max, value=min(4, k_max))
    preview_all = st.checkbox("Preview cluster sizes for this K", value=True)
    candidates = [chosen_k] if preview_all else []
else:
    candidates = sorted(set([
        max(k_min, recommended_k - 1),
        recommended_k,
        min(k_max, recommended_k + 1),
    ]))

    left_k, mid_k, right_k = candidates[0], candidates[1], candidates[2] if len(candidates) == 3 else candidates[-1]

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f"**{left_k} segments**\n\nSofter / broader groups")
    with colB:
        st.markdown(f"**{mid_k} segments ⭐**\n\nSweet spot (recommended)")
    with colC:
        st.markdown(f"**{right_k} segments**\n\nMore granular groups")

    choice = st.radio(
        "Select an option",
        options=[left_k, mid_k, right_k],
        index=1,
        format_func=lambda k: f"{k} segments" + (" ⭐ recommended" if k == recommended_k else ""),
        horizontal=True,
    )
    chosen_k = int(choice)
    preview_all = st.checkbox("Preview all recommended options", value=False)

@st.cache_data(show_spinner=False)
def quick_cluster_sizes(X_in: np.ndarray, k: int, rs: int, n_init_in: int):
    km = KMeans(n_clusters=k, random_state=rs, n_init=n_init_in)
    labels = km.fit_predict(X_in)
    counts = pd.Series(labels).value_counts().sort_index()
    return counts

if preview_all and recommended_k is not None:
    st.write("Preview: cluster size distributions for the recommended options")
    preview_cols = st.columns(len(candidates))
    for i, k in enumerate(candidates):
        counts = quick_cluster_sizes(X, int(k), int(random_state), int(n_init))
        fig_sizes = px.bar(
            x=[f"Seg {int(ix)}" for ix in counts.index],
            y=counts.values,
            title=f"Cluster sizes (K={k})",
            labels={"x": "Segment", "y": "Count"},
        )
        with preview_cols[i]:
            st.plotly_chart(fig_sizes, use_container_width=True)
elif preview_all and recommended_k is None:
    counts = quick_cluster_sizes(X, int(chosen_k), int(random_state), int(n_init))
    fig_sizes = px.bar(
        x=[f"Seg {int(ix)}" for ix in counts.index],
        y=counts.values,
        title=f"Cluster sizes (K={chosen_k})",
        labels={"x": "Segment", "y": "Count"},
    )
    st.plotly_chart(fig_sizes, use_container_width=True)


# =========================================================
# 7) Run Final Clustering
# =========================================================
st.subheader("7) Run segmentation")

run_clicked = st.button("🚀 Run Segmentation Analysis", type="primary")

@st.cache_data(show_spinner=False)
def run_kmeans(X_in: np.ndarray, k: int, rs: int, n_init_in: int):
    km = KMeans(n_clusters=k, random_state=rs, n_init=n_init_in)
    labels = km.fit_predict(X_in)
    centers = km.cluster_centers_
    inertia = float(km.inertia_)
    return labels, centers, inertia

if run_clicked:
    with st.spinner(f"Creating {chosen_k} {use_case_label}..."):
        labels, centers, inertia_val = run_kmeans(X, int(chosen_k), int(random_state), int(n_init))

        # Store ONLY labels + settings; rebuild segmented_df dynamically later
        st.session_state["segment_labels"] = labels.astype(int)
        st.session_state["segment_k"] = int(chosen_k)
        st.session_state["segment_features"] = final_features
        st.session_state["segment_primary_metric"] = primary_metric
        st.session_state["segment_use_case"] = use_case

    st.success(f"✅ Identified **{chosen_k}** distinct segments!")


# =========================================================
# 8) Display Results (only if segmentation exists)
# =========================================================
k_used = st.session_state.get("segment_k")
seg_features = st.session_state.get("segment_features")
primary_metric_used = st.session_state.get("segment_primary_metric", primary_metric)

labels = st.session_state.get("segment_labels")
if labels is None:
    st.info("Run segmentation to see results.")
    st.stop()

# SAFETY: ensure labels match current df rows
if len(labels) != len(df):
    st.error(
        "Segment labels do not match the current dataset length. "
        "This usually happens if the dataset changed after clustering. "
        "Please re-run segmentation."
    )
    st.stop()

# Rebuild from CURRENT df so engineered columns are included
segmented_df = df.copy()
segmented_df["Segment"] = labels

st.subheader("8) Results")

# 8A) Segment sizes
seg_counts = segmented_df["Segment"].value_counts().sort_index()
fig_counts = px.bar(
    x=[f"Segment {int(i)}" for i in seg_counts.index],
    y=seg_counts.values,
    title="Segment sizes",
    labels={"x": "Segment", "y": "Count"},
)
st.plotly_chart(fig_counts, use_container_width=True)

# 8A.2) Primary metric by segment
st.subheader("Primary metric by segment (Change metric insection 2 above)")
metric_means = segmented_df.groupby("Segment")[primary_metric_used].mean(numeric_only=True).reset_index()
metric_means = metric_means.sort_values(primary_metric_used, ascending=False)

fig_metric = px.bar(
    metric_means,
    x="Segment",
    y=primary_metric_used,
    title=f"Average {primary_metric_used} by segment",
)
st.plotly_chart(fig_metric, use_container_width=True)

# Silhouette (not prominent)
try:
    sil = float(silhouette_score(X, segmented_df["Segment"].values))
    st.caption(f"Quality check (silhouette score): {sil:.3f} (shown for reference only)")
except Exception:
    st.caption("Quality check: silhouette score not available for this dataset/selection.")

# 8B) Visual cluster plot (PCA if >2 features)
st.subheader("B) Segment visualization")

viz_note = ""
if seg_features is not None and len(seg_features) == 2:
    viz_df = segmented_df[[seg_features[0], seg_features[1], "Segment"]].copy()
    fig_scatter = px.scatter(
        viz_df,
        x=seg_features[0],
        y=seg_features[1],
        color="Segment",
        title="Segment plot (using your two selected features)",
    )
    viz_note = "Plot shows your actual selected features."
else:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    viz_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    viz_df["Segment"] = segmented_df["Segment"].values
    fig_scatter = px.scatter(
        viz_df,
        x="PC1",
        y="PC2",
        color="Segment",
        title="Segment plot (PCA visualization)",
    )
    viz_note = "Using PCA for visualization only — segment profiles below use your actual business features."

st.plotly_chart(fig_scatter, use_container_width=True)
st.caption(viz_note)

# 8C) Segment profiles
st.subheader("C) Segment profiles")

if seg_features is None:
    st.warning("No feature list found for segment profiling. Please re-run segmentation.")
    st.stop()

profile_df = segmented_df[seg_features + ["Segment"]].groupby("Segment").mean(numeric_only=True)
overall_avg = segmented_df[seg_features].mean(numeric_only=True)

# Sort profile table by primary metric (business-friendly)
if primary_metric_used in profile_df.columns:
    profile_df = profile_df.sort_values(primary_metric_used, ascending=False)

def diff_insights(segment_means: pd.Series, overall_means: pd.Series, threshold: float = 0.20):
    insights = []
    for feat in segment_means.index:
        seg_val = segment_means[feat]
        base = overall_means[feat]
        if pd.isna(seg_val) or pd.isna(base) or base == 0:
            continue
        pct = (seg_val - base) / abs(base)
        if abs(pct) >= threshold:
            direction = "higher" if pct > 0 else "lower"
            insights.append((feat, pct, direction))
    insights.sort(key=lambda x: abs(x[1]), reverse=True)
    return insights

# Expander per segment
for seg_id in profile_df.index.tolist():
    seg_n = int(seg_counts.loc[seg_id])
    header = f"Segment {int(seg_id)} ({seg_n:,} items)"
    with st.expander(header, expanded=False):
        st.write("**Average feature values in this segment:**")
        seg_means = profile_df.loc[seg_id]

        # Show primary metric first if present
        ordered = seg_means.copy()
        if primary_metric_used in ordered.index:
            ordered = pd.concat([ordered.loc[[primary_metric_used]], ordered.drop(primary_metric_used)])

        st.dataframe(ordered.to_frame("segment_avg"), use_container_width=True)

        st.write("**What makes this segment different?** (only showing features > 20% from overall average)")
        insights = diff_insights(seg_means, overall_avg, threshold=0.20)
        if not insights:
            st.caption("No features differ by more than 20% from the overall average based on current selection.")
        else:
            for feat, pct, direction in insights[:10]:
                flag = "⭐ " if feat == primary_metric_used else ""
                st.write(f"• {flag}**{feat}**: {abs(pct)*100:.0f}% {direction} than average")

# 8D) Download segmented data
st.subheader("D) Download")

csv_bytes = segmented_df.to_csv(index=False).encode("utf-8")
filename = f"{use_case.lower()}_segments_{int(k_used)}clusters.csv" if k_used is not None else "segmented_data.csv"
st.download_button(
    "Download segmented data as CSV",
    data=csv_bytes,
    file_name=filename,
    mime="text/csv",
)

# 8E) LLM Business Insights (stub)
st.subheader("E) AI business insights (optional)")

segment_summary_payload = {
    "use_case": use_case,
    "k": int(k_used) if k_used is not None else None,
    "features_used": seg_features,
    "primary_metric": primary_metric_used,
    "segment_sizes": seg_counts.to_dict(),
    "segment_means": profile_df.round(4).to_dict(),
    "overall_means": overall_avg.round(4).to_dict(),
    "imputation_strategy": impute_strategy,
}

ai_recommendations_ui("segment_insights", payload=segment_summary_payload)

# =========================================================
# F) Analyze segments by any feature (engineered or original)
# =========================================================
st.subheader("F) Analyze segments by a chosen feature")

feature_to_analyze = st.selectbox(
    "Pick any column to compare across segments (engineered features included)",
    options=[c for c in segmented_df.columns if c != "Segment"],
    index=0,
)

temp = segmented_df[["Segment", feature_to_analyze]].copy()
is_numeric = pd.api.types.is_numeric_dtype(temp[feature_to_analyze])
is_datetime = pd.api.types.is_datetime64_any_dtype(temp[feature_to_analyze])

if is_numeric:
    plot_rows = temp.dropna()
    if plot_rows.empty:
        st.warning("No rows left after dropping missing values for this feature.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            plot_type = st.selectbox("Plot type", ["Box + Points", "Violin + Points", "Box only"], index=0)
        with c2:
            show_points = st.checkbox("Show individual points (swarm-like)", value=True)
        with c3:
            log_scale = st.checkbox("Log scale (useful for skewed metrics)", value=False)

        order = (
            plot_rows.groupby("Segment")[feature_to_analyze]
            .median()
            .sort_values(ascending=True)
            .index.tolist()
        )
        plot_rows["Segment"] = pd.Categorical(plot_rows["Segment"], categories=order, ordered=True)

        if plot_type == "Violin + Points":
            fig = px.violin(
                plot_rows,
                x="Segment",
                y=feature_to_analyze,
                box=True,
                points="all" if show_points else False,
                title=f"Segments compared by: {feature_to_analyze}",
            )
        elif plot_type == "Box only":
            fig = px.box(
                plot_rows,
                x="Segment",
                y=feature_to_analyze,
                points=False,
                title=f"Segments compared by: {feature_to_analyze}",
            )
        else:
            fig = px.box(
                plot_rows,
                x="Segment",
                y=feature_to_analyze,
                points="all" if show_points else False,
                title=f"Segments compared by: {feature_to_analyze}",
            )

        if log_scale:
            fig.update_yaxes(type="log")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Summary by segment**")
        summary = (
            plot_rows.groupby("Segment")[feature_to_analyze]
            .agg(["count", "mean", "median", "min", "max", "std"])
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

        explain_graph_ui(
            section_key=f"seg_feature_{feature_to_analyze}",
            fig=fig,
            context={
                "chart_type": "segment_numeric_feature_comparison",
                "feature": feature_to_analyze,
                "summary": summary.to_dict(),
                "notes": "Distribution of the selected feature by segment.",
            },
        )

elif is_datetime:
    st.info("Datetime analysis is coming soon. For now, engineer time-based numeric features (e.g., Month, Days Since).")
    st.caption("Tip: Create features like `month`, `weekday`, or `days_since_last_purchase` in Feature Engineering.")
else:
    plot_rows = temp.dropna()
    if plot_rows.empty:
        st.warning("No rows left after dropping missing values for this feature.")
    else:
        top_n = st.slider("Top categories to show", 3, 25, 10)

        counts = (
            plot_rows.groupby(["Segment", feature_to_analyze])
            .size()
            .reset_index(name="count")
        )

        top_cats = (
            plot_rows[feature_to_analyze]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )

        counts["category_display"] = counts[feature_to_analyze].where(
            counts[feature_to_analyze].isin(top_cats),
            other="(Other)"
        )

        counts2 = (
            counts.groupby(["Segment", "category_display"])["count"]
            .sum()
            .reset_index()
        )

        totals = counts2.groupby("Segment")["count"].sum().rename("segment_total")
        counts2 = counts2.merge(totals, on="Segment")
        counts2["pct"] = counts2["count"] / counts2["segment_total"] * 100

        fig = px.bar(
            counts2,
            x="Segment",
            y="pct",
            color="category_display",
            barmode="stack",
            title=f"Segment composition by: {feature_to_analyze} (top {top_n} categories)",
            labels={"pct": "% within segment", "category_display": feature_to_analyze},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top categories by segment**")
        top_table = (
            counts2.sort_values(["Segment", "pct"], ascending=[True, False])
            .groupby("Segment")
            .head(5)
            .reset_index(drop=True)
        )
        st.dataframe(top_table, use_container_width=True)

        explain_graph_ui(
            section_key=f"seg_cat_{feature_to_analyze}",
            fig=fig,
            context={
                "chart_type": "segment_categorical_composition",
                "feature": feature_to_analyze,
                "top_n": top_n,
                "notes": "Stacked bar shows category mix within each segment.",
            },
        )
