import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📦 Add src/ to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# 🔹 Import custom modules
from preprocess import load_clean_data
from clustering import segment_customers
from eda import perform_eda
from pca_analysis import perform_pca, plot_pca_variance, plot_pca_clusters
from ai_insights import predict_high_spenders

# ⚙️ Streamlit Setup
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 AI-Powered Customer Segmentation Dashboard")

# -------------------- LOAD DATA --------------------
df = load_clean_data()
st.write("📌 Columns in Dataset:", df.columns.tolist())

# -------------------- FEATURE ENGINEERING --------------------
if "Year_Birth" in df.columns:
    df["Age"] = 2025 - df["Year_Birth"]
else:
    st.error("❌ 'Year_Birth' column not found. Please check your dataset format.")
    st.stop()

df["TotalSpend"] = df[[
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]].sum(axis=1)

df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["TenureDays"] = (pd.to_datetime("2025-01-01") - df["Dt_Customer"]).dt.days

st.success("✅ Data Loaded & Enriched")

# -------------------- RAW DATA VIEW --------------------
with st.expander("📄 View Raw Data"):
    st.dataframe(df.head())

# -------------------- CLUSTERING --------------------
df, _, _ = segment_customers(df)

with st.expander("📊 Clustering Insights"):
    st.write("### Cluster Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Cluster", data=df, palette="viridis", ax=ax)
    st.pyplot(fig)

    st.write("### Avg Spend per Cluster")
    cluster_spend = df.groupby("Cluster")[[
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].mean()
    st.dataframe(cluster_spend)

# -------------------- EDA --------------------
with st.expander("📈 Exploratory Data Analysis"):
    perform_eda(df)

# -------------------- PCA --------------------
with st.expander("🧩 PCA Visualization"):
    selected_features = [
        "Income", "Recency", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases",
        "NumWebVisitsMonth", "TotalSpend", "Age", "TenureDays"
    ]
    df_pca, pca_model = perform_pca(df, selected_features, n_components=2)
    df_pca, _, _ = segment_customers(df_pca)
    plot_pca_variance(pca_model)
    plot_pca_clusters(df_pca)

# -------------------- AI INSIGHTS --------------------
with st.expander("🤖 AI: Predict High Spenders"):
    df = predict_high_spenders(df)

# -------------------- FINAL DATA PREVIEW --------------------
with st.expander("🔍 Final Preview with Predictions"):
    st.dataframe(df[[
        "Income", "Recency", "NumWebPurchases", "Cluster", "SpendingPrediction"
    ]].head())
