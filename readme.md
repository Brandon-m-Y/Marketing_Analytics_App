## Marketing Segmentation App (v1.0)

This project is a **data analytics and machine learning app for customer segmentation**. It is designed to help marketers and analysts understand their customer base, identify meaningful groups (segments), and use those insights to drive targeted campaigns, product decisions, and personalization.

This is **Version 1** of the app, focused primarily on segmentation. Future versions will expand into **sales forecasting** and additional analytics capabilities.

---

## The Solution

- ✅ Runs entirely on your computer—no cloud uploads  
- ✅ No tracking, no data collection, no privacy compromises  
- ✅ Free forever—no subscriptions or usage limits  
- ✅ Open source—verify our privacy claims yourself  

---

## Core Objectives

- **Customer Understanding**: Group customers based on behavior, demographics, transactions, or engagement.
- **Actionable Segments**: Produce interpretable segment profiles (e.g. “high-value loyal”, “price-sensitive new users”).
- **Repeatable Workflow**: Provide a reusable pipeline that can be run on new customer/export data.
- **Marketing Enablement**: Make it easy to export segment labels back into CRM/BI tools.

---

## Current Features (v1.0)

### 📥 Data Intake
- Upload CSV, TXT, or Excel files  
- Auto-detect delimiters and formatting  
- Preview and validate before processing  
- Handle files up to 200MB  

### 🧹 Data Cleaning
- Remove duplicate rows  
- Handle missing values (drop, fill, interpolate)  
- Fix data type issues  
- Standardize formatting  

### 🔍 Exploratory Data Analysis (EDA)
- Automated summary statistics  
- Interactive visualizations (histograms, box plots, scatter plots)  
- Correlation heatmaps  
- Missing data analysis  
- Outlier detection  

### ⚙️ Feature Engineering
- Combine columns (concatenate, math operations)  
- Create calculated fields  
- Bin numeric values into categories  
- Date/time feature extraction  
- Custom transformations  

### 🎯 Marketing Segmentation
- K-means clustering with automatic elbow detection  
- Customer segmentation (RFM-ready)  
- Visualize segments with PCA  
- Export segmented customer lists  
- AI-powered segment explanations (optional, uses your API key)  

### 🔒 Privacy Features
- All processing happens locally  
- No internet connection required (except optional AI features)  
- No telemetry or usage tracking  
- No data stored on external servers  
- Optional AI features only send anonymized chart images, never raw data  

---

## High-Level Architecture

While file and module names may differ slightly in your environment, the app generally follows this structure:

- **Data ingestion & cleaning**  
  Load raw customer data (CSV/database), handle missing values, cast types, and engineer features.

- **Feature engineering**  
  Create model-ready features such as recency/frequency/monetary (RFM) scores, normalized numeric variables, and encoded categorical variables.

- **Segmentation model**  
  Apply unsupervised learning (e.g. `KMeans` or similar clustering algorithm) to group customers into \(k\) segments.

- **Evaluation & profiling**  
  Summarize and visualize segment differences (average spend, frequency, demographics, etc.).

- **Export & integration**  
  Save segment labels and profiles for use in dashboards, campaign tools, or downstream analysis.

---

## Typical Workflow

1. **Prepare data**: Export or collect your customer-level dataset (e.g. from CRM or transaction system).
2. **Run preprocessing**: Clean and engineer features (RFM, normalized metrics, encodings).
3. **Train segmentation model**: Fit the clustering algorithm and assign each customer to a segment.
4. **Inspect results**: View summary tables and charts by segment to understand behavior and value.
5. **Export segments**: Save labeled data for marketing campaigns, personalization, or reporting.

---

## Example Code Snippets

Below are representative Python snippets that illustrate the core ideas used in this app.  
Adapt the variable and file names to match your exact project structure.

### Data Loading and Basic Cleaning

```python
import pandas as pd

# Load raw customer data
df = pd.read_csv("data/customers.csv")

# Basic cleaning
df = df.drop_duplicates(subset="customer_id")
df = df.dropna(subset=["customer_id"])
```

### RFM Feature Engineering (Example)

```python
import pandas as pd

today = pd.Timestamp("2024-12-31")  # or df["date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_id").agg(
    recency=("last_purchase_date", lambda d: (today - d.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_value", "sum"),
).reset_index()
```

### Scaling and Clustering (KMeans Example)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

kmeans = KMeans(n_clusters=4, random_state=42)
rfm["segment"] = kmeans.fit_predict(X_scaled)
```

### Segment Profiling

```python
segment_summary = (
    rfm.groupby("segment")[["recency", "frequency", "monetary"]]
    .mean()
    .round(2)
)

print(segment_summary)
```

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup

1. **Clone the repository**
```bash
   git clone https://github.com/Brandon-m-Y/Marketing_Analytics_App
   cd bmy-analytics
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the app**
```bash
   streamlit run Home.py
```

4. **Open in your browser**
   - Streamlit will automatically open `http://localhost:8501`
   - If it doesn't, manually visit that URL

### First Time Using the App?

1. Click **📥 Data Intake** in the sidebar
2. Upload a CSV or Excel file (customer list, sales data, etc.)
3. Follow the guided workflow through cleaning, exploration, and segmentation
4. Download your results and AI-powered insights

That's it! Your data never leaves your computer.

### Optional: Virtual Environment (Recommended)

For a clean installation, use a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Current Limitations (v1.0)

- **Unsupervised only**: The current version focuses on clustering/segmentation and does not yet include predictive models for future metrics (e.g. sales).
- **Static pipeline**: Assumes periodic batch runs; real-time segmentation is out of scope for this version.
- **Generic schema**: You may need to adapt column names and feature logic to your specific business/domain.

---

## Roadmap & Future Versions

### v1.1 (Coming Soon)

- [ ] Linear regression for sales forecasting  
- [ ] Time series visualization  
- [ ] Coefficient of determination (R²) metrics  
- [ ] Download prediction results  

---

### Beyond v1.1

This repository is intended to evolve beyond segmentation and basic forecasting. Planned v2+ features include:

- **Advanced sales forecasting**  
  - Time-series models and/or regression models to forecast revenue, orders, or customer value by segment or overall.

- **Churn and retention analytics**  
  - Classification models to predict churn risk and recommend retention tactics.

- **Campaign impact analysis**  
  - Pre/post or uplift-style analysis to measure how different segments respond to campaigns.

- **Dashboard integration**  
  - Tighter integration with BI tools (e.g. Power BI, Tableau, or a lightweight web dashboard) to explore segments and forecasts interactively.

If you are reviewing this project as part of a portfolio, please note that **this is a first version** with a clear roadmap toward **richer forecasting and marketing analytics functionality**, including sales forecasting starting in **v1.1**.
