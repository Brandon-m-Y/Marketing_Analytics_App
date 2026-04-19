# 📊 BMY Analytics – Privacy-First Customer Segmentation & Marketing Intelligence

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.17%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](https://opensource.org/licenses/MIT)

**An end-to-end, privacy-first marketing analytics app that turns raw customer data into actionable segments — entirely on your own machine, with no cloud uploads, no subscriptions, and no data mining.**

🔗 [**Try the Live Demo →**](https://marketingseg.streamlit.app/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [End-to-End Workflow](#end-to-end-workflow)
   - [Step 1 — Business Understanding](#step-1--business-understanding--problem-definition)
   - [Step 2 — Data Acquisition & Inspection](#step-2--data-acquisition--initial-inspection)
   - [Step 3 — Exploratory Data Analysis](#step-3--exploratory-data-analysis-eda)
   - [Step 4 — Data Preprocessing & Feature Engineering](#step-4--data-preprocessing--feature-engineering)
   - [Step 5 — K-Means Clustering & Segmentation](#step-5--k-means-clustering--segmentation)
   - [Step 6 — Model Evaluation & Cluster Profiling](#step-6--model-evaluation--cluster-profiling)
   - [Step 7 — Insights & Marketing Recommendations](#step-7--insights--marketing-recommendations)
   - [Step 8 — Streamlit App & Deployment](#step-8--streamlit-app--deployment)
3. [App Features at a Glance](#app-features-at-a-glance)
4. [Tech Stack](#tech-stack)
5. [Quick Start](#quick-start)
6. [Privacy Architecture](#privacy-architecture)
7. [Roadmap](#roadmap)
8. [About the Author](#about-the-author)

---

## Project Overview

Small businesses and marketing agencies are sitting on goldmines of customer data — transaction histories, email lists, CRM exports — but lack the in-house data science capability to turn that data into decisions. Enterprise platforms like Salesforce and HubSpot charge thousands per month for segmentation features, and they do it by processing your most sensitive asset (your customer list) on their servers. BMY Analytics was built to change that.

This app delivers professional-grade customer segmentation, exploratory data analysis, and feature engineering through a clean, guided Streamlit interface — with every computation running locally on the user's machine. There are no cloud uploads, no API calls involving raw customer data, no subscriptions, and no tracking. The open-source codebase acts as a verifiable privacy guarantee.

The core analytical engine uses **K-Means clustering** paired with automatic elbow detection (via `kneed`) and silhouette scoring to find the natural groupings in any customer dataset. Segments are profiled against a user-selected primary metric and visualized with PCA scatter plots, so marketers can immediately see which segments represent their VIPs, at-risk churners, and high-potential growth opportunities — then export labeled data straight into their CRM or email platform.

The target audience is small-to-medium businesses, marketing agencies, and consultants who need repeatable, explainable segmentation they can run on fresh exports without a data science team. This is **Version 1.0**, focused on segmentation; sales forecasting and churn prediction are on the roadmap for v1.1+.

---

## End-to-End Workflow

### Step 1 — Business Understanding & Problem Definition

Most marketing budgets are wasted treating every customer identically. A loyalty offer sent to someone who already buys weekly is money left on the table; a win-back campaign sent to an active buyer creates confusion. The fundamental business question this app answers is:

> **Who are my customers, how are they different from each other, and what should I do differently for each group?**

Concretely, the app targets four outcomes that drive real marketing ROI:

| Business Goal | Analytical Lever |
|---|---|
| Identify VIP customers | High-monetary, high-frequency segment |
| Win back churning customers | High-recency (days since purchase), low-frequency segment |
| Find upsell opportunities | Mid-value segments with upward spend trends |
| Optimize campaign spend | Suppress low-engagement segments from broad campaigns |

---

### Step 2 — Data Acquisition & Initial Inspection

The app accepts any structured customer dataset — CSV, TXT, or Excel — with automatic delimiter detection and format validation. Files up to 200 MB are supported. After upload, a preview pane shows row/column counts, detected types, and a quick data health summary so users can assess quality before any processing begins.

![BMY Analytics Home Page — Privacy-first value proposition and guided workflow overview](images/Home%20Page.PNG)

*The home screen frames the privacy guarantee upfront and routes users through a four-step workflow: Upload → Clean → Analyze → Act.*

---

![Data Intake Page — Upload CSV/Excel, auto-detect delimiters, preview and validate before processing](images/Load%20Your%20Data.PNG)

*The Data Intake page supports CSV, TXT, and Excel files up to 200 MB. Delimiter auto-detection and an instant data preview let users validate structure before committing to the pipeline.*

---

### Step 3 — Exploratory Data Analysis (EDA)

Before any model is fit, the EDA module builds a comprehensive picture of the dataset's shape, distribution, and internal relationships. All charts are rendered with Plotly for interactivity — hover tooltips, zoom, and pan work out of the box.

**A) Data Overview Metrics**
Row count, column count, missing cell totals, duplicate row count, and memory footprint are surfaced as headline metrics immediately.

**B) Missingness Analysis**
A ranked bar chart shows the top N columns by missing percentage alongside raw missing counts. Columns with >30% missingness are flagged so users can decide whether to impute, drop, or investigate upstream before moving to modeling.

**C) Histogram & Distribution Explorer**
Users select any numeric column to generate a histogram (bin count adjustable via slider) alongside summary stats (mean, std, unique count). A companion boxplot renders below to highlight median, IQR spread, and outlier candidates simultaneously.

**D) Scatter Plot Explorer**
A point-and-click scatter builder lets users place any two numeric columns on X and Y axes, with an optional third dimension for color-coding by any column (numeric or categorical). This is the fastest way to spot linear relationships, clusters, or outliers before the formal segmentation step.

**E) Correlation Heatmap**
A Pearson correlation matrix is computed across all numeric columns and rendered as an interactive heatmap. Strong positive correlations (close to +1) highlight features that move together — which matters for deciding whether to include both in a clustering model or collapse them into a single composite score.

![EDA Page — Interactive correlation heatmap, histograms, scatter explorer, and missingness analysis](images/Explore%20Your%20Data.PNG)

*The EDA module surfaces distribution shapes, pairwise correlations, and data quality gaps through interactive Plotly charts — no code required. The correlation heatmap is especially useful for identifying multicollinear features before clustering.*

---

### Step 4 — Data Preprocessing & Feature Engineering

Raw CRM or transaction exports rarely arrive model-ready. The Feature Engineering page bridges the gap between raw columns and model-ready inputs without requiring SQL or Python.

Supported transformations include:
- **Calculated fields** — Combine any two numeric columns with arithmetic operators (+, −, ×, ÷) to derive composite metrics (e.g., `revenue_per_order = total_revenue / order_count`)
- **Column concatenation** — Merge text columns for downstream categorical groupings
- **Numeric binning** — Convert continuous values into labeled buckets (e.g., bin `days_since_purchase` into `["< 30 days", "30–90 days", "> 90 days"]`)
- **Date/time extraction** — Parse timestamp columns into month, weekday, quarter, or "days since" features that K-Means can consume directly
- **Custom transformations** — Standardize formatting, cast types, or apply user-defined expressions

For RFM-style segmentation, the recommended flow is to engineer Recency, Frequency, and Monetary columns before entering the segmentation module:

```python
today = pd.Timestamp("today")

rfm = df.groupby("customer_id").agg(
    recency=("last_purchase_date", lambda d: (today - d.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_value", "sum"),
).reset_index()
```

Internally, the segmentation module handles `StandardScaler` normalization and `SimpleImputer` missing-value fill (mean, median, or most-frequent strategies are all selectable) so that no single high-magnitude feature dominates cluster formation.

![Feature Engineering Page — Derived columns, binning, date extraction, and calculated field builder](images/Engineer%20New%20Features.PNG)

*The Feature Engineering page lets marketers build RFM scores, date-derived features, and composite metrics without writing code. Engineered columns flow automatically into the EDA and Segmentation modules downstream.*

---

![Data Cleaning Page — Deduplication, missing value handling, type fixing, and standardization tools](images/Clean%20Your%20Data.PNG)

*The Data Cleaning page resolves the most common data quality issues in customer exports: duplicate rows, missing values (drop/fill/interpolate), type mismatches, and inconsistent formatting — all before any analysis begins.*

---

### Step 5 — K-Means Clustering & Segmentation

The segmentation engine is built around `sklearn.cluster.KMeans`, with two key quality-of-life additions that make the output actionable for non-technical marketers: **automatic elbow detection** and **guided K selection**.

**Automatic Feature Preparation**
Before fitting any model, the app auto-filters the feature set to remove:
- ID-like columns (name contains "id" and all values are unique)
- Zero-variance columns (constant value across all rows)
- Sequential row-number columns (detected via a diff-based heuristic)

This prevents meaningless dimensions from distorting cluster geometry. Users can override auto-selection and hand-pick features in the Advanced Options panel.

**Elbow Analysis**
The app iterates K-Means across a user-defined K range (default 2–10) and plots inertia (within-cluster sum of squared distances) against K. The `kneed` library applies a convex curve detector to identify the inflection point — the K where adding more clusters stops meaningfully reducing inertia.

**Guided K Selection**
Rather than presenting a raw slider, the interface shows three candidate values (elbow − 1, elbow, elbow + 1) side by side with a recommended marker. Users can preview cluster size distributions for all three options before committing to a final K.

**Scaling & Fitting**
Features are imputed then normalized via `StandardScaler` before fitting, ensuring that high-magnitude features (e.g., total revenue) don't overshadow low-magnitude ones (e.g., days since purchase). The final model uses `n_init=10` random restarts and a configurable random seed for reproducibility.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_imputed = imputer.fit_transform(features_df)
X_scaled  = scaler.fit_transform(X_imputed)

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

![Customer Segmentation Page — K-Means clustering with elbow detection, PCA visualization, and segment profiling](images/Segment%20Your%20Customers.PNG)

*The segmentation page walks users from raw feature selection through elbow analysis, guided K selection, PCA scatter visualization, and per-segment profiling — outputting a labeled CSV ready for CRM import.*

---

### Step 6 — Model Evaluation & Cluster Profiling

**Silhouette Score**
After fitting, a silhouette score is computed using `sklearn.metrics.silhouette_score`. Values approaching +1 indicate dense, well-separated clusters; values near 0 suggest overlapping segments; negative values signal misassignment. The score is surfaced as a quality reference rather than a hard gate, since business interpretability often matters more than geometric purity.

**Cluster Size Distribution**
A bar chart shows customer counts per segment immediately after fitting, flagging any severely imbalanced clusters (e.g., one segment capturing 90% of customers) that would undermine targeting value.

**Segment Profiles**
Each segment is expanded into a profile table showing average values for every clustering feature, sorted by the user-selected primary metric (e.g., average revenue). A diff-insight engine highlights any feature where the segment deviates more than 20% from the overall dataset mean — surfacing the characteristics that actually distinguish the group.

**PCA Scatter Visualization**
For datasets with more than 2 features, a two-component PCA projection renders all data points color-coded by segment, giving an intuitive visual confirmation of how separable the clusters are in reduced-dimensional space.

**Per-Feature Segment Comparison**
A flexible analysis panel lets users select any column — including engineered features — and compare its distribution across segments via box plots, violin plots, or stacked bar charts (for categorical columns). Summary statistics (count, mean, median, min, max, std) render as a table alongside each chart.

---

### Step 7 — Insights & Marketing Recommendations

Once segments are profiled, the business interpretation layer translates statistical output into campaign-ready language. The diff-insight engine generates plain-English callouts like:

> • **monetary**: 85% higher than average — *this segment is your top revenue driver*
> • **recency**: 120% higher than average — *these customers haven't purchased in a long time; prime win-back targets*
> • **frequency**: 40% lower than average — *occasional buyers with upsell potential*

Typical segment archetypes that emerge from RFM-style clustering:

| Segment Profile | Marketing Action |
|---|---|
| **VIPs** — high frequency, high monetary, low recency | Loyalty rewards, early access, referral programs |
| **At-Risk** — historically high value, high recency (lapsed) | Win-back emails, reactivation discounts, personal outreach |
| **Promising** — moderate frequency, rising monetary | Upsell to premium tier, cross-sell complementary products |
| **Price-Sensitive** — high frequency, low monetary | Bundle deals, bulk discounts, subscription offers |
| **New/Explorers** — low frequency, low monetary | Onboarding sequences, first-purchase incentives |

Segmented data is exportable as a labeled CSV (one row per customer with a `Segment` column appended) for direct import into Mailchimp, Klaviyo, HubSpot, or any CRM that accepts flat files.

An optional AI-powered insights feature (coming in v1.1) will send anonymized segment summaries — never raw customer data — to an LLM endpoint using the user's own API key, returning plain-English segment names and campaign strategy recommendations.

---

### Step 8 — Streamlit App & Deployment

The app is structured as a multi-page Streamlit application with a persistent `session_state` pipeline so that cleaned and engineered data flows forward automatically through each module without re-uploading.

```
app.py                         ← Home / landing page
pages/
  1_📥_Data_Intake.py          ← Upload, preview, validate
  2_🧹_Data_Cleaning.py        ← Dedup, impute, type-fix
  3_🧠_Feature_Engineering.py  ← Derived columns, binning, date features
  4_🔎_EDA.py                  ← Distributions, correlations, scatter
  5_🧩_Customer_Segmentation.py← K-Means, elbow, PCA, profiles, export
src/
  loaders.py                   ← File parsing utilities
```

The production deployment runs on **Streamlit Community Cloud** with zero server-side data persistence — all session state is ephemeral and local to the browser session. The app can also be run fully offline after a one-time `pip install`.

---

## App Features at a Glance

| Module | Key Capabilities |
|---|---|
| 📥 Data Intake | CSV / TXT / Excel upload, auto-delimiter detection, 200 MB limit, instant preview |
| 🧹 Data Cleaning | Deduplication, missing value handling (drop/fill/interpolate), type casting, format standardization |
| 🧠 Feature Engineering | Calculated fields, binning, date/time extraction, column concatenation, custom transformations |
| 🔎 EDA | Missingness heatmap, histogram + boxplot explorer, scatter builder, Pearson correlation matrix |
| 🧩 Customer Segmentation | K-Means clustering, auto elbow detection, silhouette scoring, PCA visualization, segment profiling, CSV export |
| 🔒 Privacy | 100% local processing, no telemetry, no cloud uploads, works offline |
| 🤖 AI Insights | Optional — anonymized chart summaries only, requires user-provided API key (v1.1) |

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| App framework | Streamlit | ≥ 1.54 |
| Data manipulation | pandas, NumPy | ≥ 2.0, ≥ 1.24 |
| Machine learning | scikit-learn | ≥ 1.3 |
| Statistical computing | SciPy | ≥ 1.11 |
| Elbow detection | kneed | ≥ 0.8.5 |
| Visualization | Plotly, Seaborn, Matplotlib | ≥ 5.17, ≥ 0.12, ≥ 3.7 |
| File I/O | openpyxl, pyarrow | ≥ 3.1, ≥ 18 |
| Environment | python-dotenv | ≥ 1.0 |

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Brandon-m-Y/Marketing_Analytics_App
cd Marketing_Analytics_App

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Streamlit will open `http://localhost:8501` automatically. If it doesn't, navigate there manually.

### First Run

1. Click **📥 Data Intake** in the sidebar
2. Upload any customer CSV or Excel file (transaction history, CRM export, email list with metadata)
3. Clean → Engineer → Explore → Segment — each page passes data forward automatically
4. Download your labeled segment file and import it into your CRM or email platform

Your data never leaves your computer.

### Sample Dataset

A sample marketing campaign dataset (`marketing_campaign_data.csv`) is included in the repository so you can explore the full workflow without needing your own data first.

---

## Privacy Architecture

BMY Analytics is built on a privacy-first architecture as a first principle, not an afterthought.

| Property | Implementation |
|---|---|
| **No server** | Streamlit Community Cloud serves only static app code; all computation runs in the user's browser session |
| **No persistence** | Session state is ephemeral — data is cleared when the browser tab closes |
| **No telemetry** | Zero analytics, tracking pixels, or usage logging in the codebase |
| **No data egress** | Raw customer data is never serialized or sent anywhere — the optional AI feature sends only anonymized chart images |
| **Open source** | Every claim above is verifiable by reading the code |

For organizations with strict compliance requirements (GDPR, CCPA, HIPAA), local execution means customer PII never crosses a network boundary under this tool's operation.

---

## Roadmap

### v1.1 — Coming Soon
- [ ] Linear regression for sales forecasting
- [ ] Time-series visualization (trend lines, seasonality decomposition)
- [ ] Coefficient of determination (R²) and RMSE metrics
- [ ] AI-powered segment naming and campaign recommendations (user's API key, anonymized summaries only)
- [ ] Downloadable forecast results

### v2.0 — Planned
- [ ] Churn prediction (classification models with probability scores)
- [ ] Campaign uplift / A/B impact analysis
- [ ] Agglomerative clustering as an alternative to K-Means
- [ ] Power BI / Tableau export connectors
- [ ] Real-time scoring mode for new customer records

---

## About the Author

**Brandon Ytuarte** — Data Analyst & Aspiring Data Scientist

BMY Analytics is a portfolio project demonstrating an end-to-end data product built for real small-business use cases: privacy-respecting, open-source, and free.

[LinkedIn](https://www.linkedin.com/in/brandon-m-ytuarte/) • [GitHub](https://github.com/Brandon-m-Y) • [Live App](https://marketingseg.streamlit.app/)

Questions, feedback, or feature requests? [Open an issue](https://github.com/Brandon-m-Y/Marketing_Analytics_App/issues) or message me on LinkedIn — this is a living project and user input directly shapes the roadmap.

---

*MIT License · Privacy-First · Open Source · Free Forever*
