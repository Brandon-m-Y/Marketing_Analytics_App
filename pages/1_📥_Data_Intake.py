import streamlit as st
from src.loaders import load_csv, load_excel

st.set_page_config(
    page_title="Data Intake",
    page_icon="📥",
    layout="wide"
)

st.title("📥 Data Intake")
st.write("Upload your dataset and confirm formatting before running EDA.")

# ────────────────────────────────────────────────
#   Cache loaders (unchanged)
# ────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading file...")
def cached_load_csv(file_bytes: bytes, delimiter):
    return load_csv(file_bytes, delimiter)

@st.cache_data(show_spinner="Loading file...")
def cached_load_excel(file_bytes: bytes):
    return load_excel(file_bytes)

delim_map = {
    "Auto-detect": None,
    ",": ",",
    ";": ";",
    "|": "|",
    "Tab": "\t"
}

# ────────────────────────────────────────────────
#   Sidebar – upload & controls
# ────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Settings")
    uploaded = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx", "xls"])

    delimiter_choice = "Auto-detect"
    if uploaded is not None and uploaded.name.lower().endswith((".csv", ".txt")):
        delimiter_choice = st.selectbox("Delimiter", options=list(delim_map.keys()), index=0)

    col_load, col_clear = st.columns([3, 2])
    with col_load:
        load_clicked = st.button("Load file", type="primary")
    with col_clear:
        if st.button("Clear data"):
            for key in ["raw_df", "file_name", "loaded_file_hash"]:
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.rerun()

# ────────────────────────────────────────────────
#   Loading logic  →  only runs on button press
# ────────────────────────────────────────────────

if load_clicked and uploaded is not None:
    file_bytes = uploaded.getvalue()
    filename = uploaded.name.lower()

    try:
        if filename.endswith((".csv", ".txt")):
            df = cached_load_csv(file_bytes, delim_map[delimiter_choice])
        else:
            df = cached_load_excel(file_bytes)

        st.session_state["raw_df"] = df
        st.session_state["file_name"] = uploaded.name
        # Optional: helps detect file change
        st.session_state["loaded_file_hash"] = hash(uploaded.getvalue())

        st.success(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    except Exception as e:
        st.session_state.pop("raw_df", None)
        st.error(f"Could not load file: {e}")

# ────────────────────────────────────────────────
#   Show content whenever we have data in session state
# ────────────────────────────────────────────────

df = st.session_state.get("raw_df")

if df is None:
    if uploaded is None:
        st.info("Upload a dataset using the sidebar.")
    else:
        st.warning("Click **Load file** in the sidebar to parse the uploaded dataset.")
    st.stop()

# ── From here on we have a valid DataFrame ──

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

st.subheader("Basic summary")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

st.subheader("Summary Statistics of Your Data")
st.dataframe(df.describe())

# Seamless flow into next page
st.divider()
st.subheader("Next step")
st.write("Proceed to EDA to explore distributions, missingness, correlations, and relationships.")

if st.button("➡️ Continue to Data Cleaning"):
    st.switch_page("pages/2_🧹_Data_Cleaning.py")   # use / not \ on all platforms
