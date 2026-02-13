import streamlit as st
import pandas as pd
import numpy as np

st.title("🧹 Data Cleaning")
st.write("Review suggested type conversions and create a cleaned dataset for EDA.")

# -----------------------------
# Require data from Intake
# -----------------------------
raw_df = st.session_state.get("raw_df")
if raw_df is None:
    st.warning("No dataset found. Please upload one in **📥 Data Intake** first.")
    if st.button("➡️ Go to Data Intake"):
        st.switch_page("pages/1_📥_Data_Intake.py")
    st.stop()

st.caption(f"File: {st.session_state.get('file_name', 'Unknown')}")

# -----------------------------
# Helpers for suggestions
# -----------------------------
YES_NO_SETS = [
    {"yes", "no"},
    {"y", "n"},
    {"true", "false"},
    {"t", "f"},
    {"1", "0"},
    {"male", "female"},
    {"m", "f"},
]

def is_binary_like(series: pd.Series) -> bool:
    vals = (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    if len(vals) != 2:
        return False
    vset = set(vals)
    return any(vset == s for s in YES_NO_SETS)

def suggest_column_action(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n = len(s)
    non_null = s.dropna()

    suggestion = {
        "column": col,
        "current_dtype": str(s.dtype),
        "suggested_action": "none",
        "reason": "",
        "details": "",
    }

    if n == 0:
        return suggestion

    # Already datetime?
    if pd.api.types.is_datetime64_any_dtype(s):
        suggestion["suggested_action"] = "keep_datetime"
        suggestion["reason"] = "Already datetime"
        return suggestion

    # Numeric?
    if pd.api.types.is_numeric_dtype(s):
        # Sometimes numeric columns are actually codes/categories
        nunique = non_null.nunique()
        if nunique <= max(20, int(0.01 * n)):
            suggestion["suggested_action"] = "maybe_categorical"
            suggestion["reason"] = "Low unique count for a numeric column"
            suggestion["details"] = f"unique={nunique}"
        else:
            suggestion["suggested_action"] = "keep_numeric"
            suggestion["reason"] = "Already numeric"
        return suggestion

    # Object / string-like
    as_str = non_null.astype(str).str.strip()
    nunique = as_str.nunique()

    # Try date parse success rate
    parsed = pd.to_datetime(as_str, errors="coerce", infer_datetime_format=True)
    date_success = parsed.notna().mean() if len(as_str) else 0.0

    if date_success >= 0.80 and nunique > 2:
        suggestion["suggested_action"] = "convert_to_datetime"
        suggestion["reason"] = "High datetime parse success rate"
        suggestion["details"] = f"parse_success={date_success:.0%}"
        return suggestion

    # Binary-like?
    if is_binary_like(s):
        suggestion["suggested_action"] = "convert_to_binary"
        suggestion["reason"] = "Detected two-level yes/no style values"
        suggestion["details"] = f"unique={nunique}"
        return suggestion

    # Likely categorical if low cardinality
    if nunique <= 20:
        suggestion["suggested_action"] = "one_hot_encode"
        suggestion["reason"] = "Low-cardinality categorical column"
        suggestion["details"] = f"unique={nunique}"
        return suggestion

    # High-cardinality text
    if nunique > 20:
        suggestion["suggested_action"] = "keep_as_text"
        suggestion["reason"] = "High-cardinality text column"
        suggestion["details"] = f"unique={nunique}"
        return suggestion

    return suggestion

def apply_binary_mapping(series: pd.Series) -> pd.Series:
    # Normalize
    s = series.astype("string").str.strip().str.lower()

    # Common mappings
    mapping_candidates = [
        ({"yes": 1, "no": 0}, {"yes", "no"}),
        ({"y": 1, "n": 0}, {"y", "n"}),
        ({"true": 1, "false": 0}, {"true", "false"}),
        ({"t": 1, "f": 0}, {"t", "f"}),
        ({"male": 1, "female": 0}, {"male", "female"}),
        ({"m": 1, "f": 0}, {"m", "f"}),
        ({"1": 1, "0": 0}, {"1", "0"}),
    ]

    uniq = set(s.dropna().unique().tolist())
    for mapping, keyset in mapping_candidates:
        if uniq == keyset:
            return s.map(mapping).astype("Int64")

    # Fallback: factorize to 0/1 with stable ordering
    codes, _ = pd.factorize(s)
    out = pd.Series(codes, index=series.index)
    out = out.replace({-1: pd.NA}).astype("Int64")
    return out

# -----------------------------
# Build suggestions table
# -----------------------------
df = raw_df.copy()
suggestions = [suggest_column_action(df, c) for c in df.columns]
sugg_df = pd.DataFrame(suggestions)

st.subheader("Suggested conversions")
st.dataframe(sugg_df, use_container_width=True)

st.subheader("Choose what to apply")
st.write("Select conversions to apply. This will create a new dataset stored as `clean_df` for EDA and modeling.")

# Multi-select recommended columns by type
date_cols = sugg_df.loc[sugg_df["suggested_action"] == "convert_to_datetime", "column"].tolist()
bin_cols  = sugg_df.loc[sugg_df["suggested_action"] == "convert_to_binary", "column"].tolist()
ohe_cols  = sugg_df.loc[sugg_df["suggested_action"] == "one_hot_encode", "column"].tolist()

c1, c2, c3 = st.columns(3)

with c1:
    selected_date_cols = st.multiselect("Convert to datetime", df.columns.tolist(), default=date_cols)

with c2:
    selected_bin_cols = st.multiselect("Convert to binary (0/1)", df.columns.tolist(), default=bin_cols)

with c3:
    selected_ohe_cols = st.multiselect("One-hot encode (dummy vars)", df.columns.tolist(), default=ohe_cols)

st.divider()

with st.expander("Preview cleaned result (before applying)"):
    st.write("This is your raw data preview. Apply conversions below to generate `clean_df`.")
    st.dataframe(df.head(25), use_container_width=True)

# -----------------------------
# Apply conversions
# -----------------------------
apply_clicked = st.button("✅ Apply conversions and create clean_df", type="primary")

if apply_clicked:
    clean_df = df.copy()
    log = []

    # Dates
    for col in selected_date_cols:
        try:
            clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce", infer_datetime_format=True)
            log.append(f"Converted to datetime: {col}")
        except Exception as e:
            log.append(f"FAILED datetime: {col} ({e})")

    # Binary
    for col in selected_bin_cols:
        try:
            clean_df[col] = apply_binary_mapping(clean_df[col])
            log.append(f"Converted to binary: {col}")
        except Exception as e:
            log.append(f"FAILED binary: {col} ({e})")

    # One-hot
    if selected_ohe_cols:
        try:
            # Use string dtype for stable dummy creation
            tmp = clean_df[selected_ohe_cols].astype("string").fillna("⟂ Missing")
            dummies = pd.get_dummies(tmp, prefix=selected_ohe_cols, prefix_sep="=", drop_first=False)
            clean_df = clean_df.drop(columns=selected_ohe_cols).join(dummies)
            log.append(f"One-hot encoded: {', '.join(selected_ohe_cols)}")
        except Exception as e:
            log.append(f"FAILED one-hot: {selected_ohe_cols} ({e})")

    st.session_state["clean_df"] = clean_df
    st.session_state["cleaning_log"] = log

    st.success(f"Created clean_df: {clean_df.shape[0]:,} rows × {clean_df.shape[1]:,} columns")

    with st.expander("Cleaning log"):
        for line in log:
            st.write("• " + line)

    st.subheader("Preview clean_df")
    st.dataframe(clean_df.head(50), use_container_width=True)

    # st.divider()
    # if st.button("➡️ Continue to EDA"):
    #     st.switch_page("pages/3_🔎_EDA.py")

# If a clean_df already exists, show it
existing = st.session_state.get("clean_df")
if existing is not None and not apply_clicked:
    st.info("A cleaned dataset already exists (clean_df). You can re-apply conversions to overwrite it.")
    st.subheader("Current clean_df preview")
    st.dataframe(existing.head(25), use_container_width=True)

if st.button("➡️ Continue to Feature Engineering"):
        st.switch_page("pages/3_🧠_Feature_Engineering.py")