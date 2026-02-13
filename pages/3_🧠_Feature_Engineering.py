import streamlit as st
import pandas as pd
import numpy as np

st.title("🧠 Feature Engineering Toolkit")
st.write("Create new features to improve clustering and modeling performance.")

# =========================================================
# Load Data (PRIORITY: fe_df → clean_df → raw_df)
# =========================================================
fe_df = st.session_state.get("fe_df")
clean_df = st.session_state.get("clean_df")
raw_df = st.session_state.get("raw_df")

if fe_df is not None:
    df = fe_df
    source_label = "Feature Engineered Data"
elif clean_df is not None:
    df = clean_df.copy()
    st.session_state["fe_df"] = df
    source_label = "Cleaned Data"
elif raw_df is not None:
    df = raw_df.copy()
    st.session_state["fe_df"] = df
    source_label = "Raw Data"
else:
    st.warning("Please upload and clean your data first.")
    st.stop()

# Initialize feature history if needed
if "feature_history" not in st.session_state:
    st.session_state["feature_history"] = []

st.caption(f"Using: **{source_label}** | {df.shape[0]:,} rows × {df.shape[1]:,} columns")

st.divider()

# =========================================================
# Feature Type Selector
# =========================================================
feature_type = st.selectbox(
    "Select Feature Type",
    [
        "Combine Text Columns",
        "Math Operations",
        "Binning / Categories",
        "Rename Column"
    ]
)

st.divider()

# =========================================================
# 1️⃣ Combine Text Columns
# =========================================================
if feature_type == "Combine Text Columns":

    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    selected_cols = st.multiselect("Select text columns to combine", text_cols)

    separator = st.selectbox("Separator", ["Space", "Comma", "Hyphen", "None"])
    sep_map = {"Space": " ", "Comma": ", ", "Hyphen": "-", "None": ""}

    new_col_name = st.text_input("New column name")

    if selected_cols and new_col_name:

        preview = df[selected_cols].astype(str).agg(sep_map[separator].join, axis=1).head(5)

        st.subheader("Preview")
        st.dataframe(preview.to_frame(new_col_name))

        if st.button("Create Feature"):
            df[new_col_name] = df[selected_cols].astype(str).agg(sep_map[separator].join, axis=1)

            st.session_state["feature_history"].append({
                "name": new_col_name,
                "type": "Text Combination",
                "source": selected_cols
            })

            st.session_state["fe_df"] = df
            st.success(f"Created new feature: {new_col_name}")
            st.rerun()

# =========================================================
# 2️⃣ Math Operations
# =========================================================
elif feature_type == "Math Operations":

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    selected_cols = st.multiselect("Select numeric columns", numeric_cols)

    operation = st.selectbox(
        "Operation",
        ["Add (+)", "Subtract (-)", "Multiply (×)", "Divide (÷)", "Average"]
    )

    new_col_name = st.text_input("New column name")

    if selected_cols and new_col_name:

        result = None

        if operation == "Add (+)":
            result = df[selected_cols].sum(axis=1)

        elif operation == "Subtract (-)":
            result = df[selected_cols[0]]
            for col in selected_cols[1:]:
                result -= df[col]

        elif operation == "Multiply (×)":
            result = df[selected_cols].prod(axis=1)

        elif operation == "Divide (÷)":
            if len(selected_cols) == 2:
                result = df[selected_cols[0]] / df[selected_cols[1]].replace(0, np.nan)
            else:
                st.warning("Division requires exactly 2 columns.")

        elif operation == "Average":
            result = df[selected_cols].mean(axis=1)

        if result is not None:
            preview_df = df[selected_cols].head(5).copy()
            preview_df["Result"] = result.head(5)

            st.subheader("Preview")
            st.dataframe(preview_df)

            if st.button("Create Feature"):
                df[new_col_name] = result

                st.session_state["feature_history"].append({
                    "name": new_col_name,
                    "type": "Math Operation",
                    "source": selected_cols,
                    "operation": operation
                })

                st.session_state["fe_df"] = df
                st.success(f"Created new feature: {new_col_name}")
                st.rerun()

# =========================================================
# 3️⃣ Binning / Categories
# =========================================================
elif feature_type == "Binning / Categories":

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    col = st.selectbox("Select numeric column", numeric_cols)

    method = st.selectbox("Binning Method", ["Equal Width", "Quantiles"])
    bins = st.slider("Number of bins", 2, 10, 4)

    new_col_name = st.text_input("New column name")

    if col and new_col_name:

        if method == "Equal Width":
            preview = pd.cut(df[col], bins=bins)
        else:
            preview = pd.qcut(df[col], q=bins, duplicates="drop")

        st.subheader("Preview")
        st.dataframe(preview.head().to_frame(new_col_name))

        if st.button("Create Feature"):
            df[new_col_name] = preview

            st.session_state["feature_history"].append({
                "name": new_col_name,
                "type": "Binning",
                "source": col,
                "bins": bins,
                "method": method
            })

            st.session_state["fe_df"] = df
            st.success(f"Created new feature: {new_col_name}")
            st.rerun()

# =========================================================
# 4️⃣ Rename Column
# =========================================================
elif feature_type == "Rename Column":

    col = st.selectbox("Select column to rename", df.columns.tolist())
    new_name = st.text_input("New column name")

    if col and new_name:

        if st.button("Rename Column"):
            df.rename(columns={col: new_name}, inplace=True)

            st.session_state["feature_history"].append({
                "name": new_name,
                "type": "Rename",
                "source": col
            })

            st.session_state["fe_df"] = df
            st.success(f"Renamed column: {col} → {new_name}")
            st.rerun()

# =========================================================
# Feature History Section
# =========================================================
st.divider()
st.subheader("Created Features")

if st.session_state["feature_history"]:
    for feat in st.session_state["feature_history"]:
        st.write(f"✓ {feat['name']} ({feat['type']})")

    if st.button("Undo Last Feature"):
        last = st.session_state["feature_history"].pop()
        if last["type"] != "Rename":
            df.drop(columns=[last["name"]], inplace=True, errors="ignore")

        st.session_state["fe_df"] = df
        st.success("Last feature removed.")
        st.rerun()

else:
    st.caption("No features created yet.")

# =========================================================
# Preview Dataset
# =========================================================
st.divider()
st.subheader("Feature-Enhanced Dataset Preview")
st.dataframe(df.head(25), use_container_width=True)

# =========================================================
# Navigation
# =========================================================
st.divider()
if st.button("➡️ Continue to EDA"):
    st.switch_page("pages/4_🔎_EDA.py")
