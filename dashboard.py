import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add src path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from preprocess import load_clean_data
from clustering import segment_customers
from ai_insights import predict_high_spenders

# --- Load & Prepare Data ---
st.title("🧠 AI-Powered Customer Segmentation Dashboard")
df = load_clean_data()
df["Age"] = 2025 - df["Year_Birth"]
df["TotalSpend"] = df[[
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]].sum(axis=1)
df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["TenureDays"] = (pd.to_datetime("2025-01-01") - df["Dt_Customer"]).dt.days

st.success("✅ Data Loaded & Preprocessed")

# --- Raw Data Preview ---
with st.expander("📌 View Raw Data"):
    st.dataframe(df.head())

# --- Clustering ---

clustered_result = segment_customers(df)
if isinstance(clustered_result, tuple):
    df = clustered_result[0]
else:
    df = clustered_result


st.subheader("📊 Clustering Insights")
st.markdown("### Cluster Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="Cluster", data=df, palette="viridis", ax=ax1)
st.pyplot(fig1)

st.markdown("### Average Spend per Cluster")
st.dataframe(df.groupby("Cluster")[[
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]].mean().reset_index())

# --- AI Insights ---
df = predict_high_spenders(df)
with st.expander("🔍 Final Preview with Predictions"):
    st.dataframe(df[["Income", "Recency", "NumWebPurchases", "Cluster", "SpendingPrediction"]].head())

# --- PCA Plot ---
st.subheader("🧬 PCA Visualization")
pca_features = [
    "Income", "Recency", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumDealsPurchases", "NumWebVisitsMonth",
    "TotalSpend", "Age", "TenureDays"
]
scaler = StandardScaler()
scaled = scaler.fit_transform(df[pca_features])
pca = PCA(n_components=2)
pcs = pca.fit_transform(scaled)
df["PC1"] = pcs[:, 0]
df["PC2"] = pcs[:, 1]

fig2, ax2 = plt.subplots()
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=df, palette="tab10", ax=ax2)
ax2.set_title("PCA Scatterplot with Clusters")
st.pyplot(fig2)

# --- EDA Section ---
with st.expander("📊 Explore EDA Insights"):
    st.markdown("### Distributions")
    for col in ["Income", "Age", "TotalSpend", "Recency", "TenureDays"]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    st.markdown("### Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[pca_features].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    if "Cluster" in df.columns:
        st.markdown("### Spend by Cluster")
        for col in ["Income", "TotalSpend", "Recency"]:
            fig, ax = plt.subplots()
            sns.boxplot(x="Cluster", y=col, data=df, ax=ax)
            st.pyplot(fig)

# --- Download Button ---
st.subheader("📥 Download Final Dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Clustered + Predicted Data", data=csv, file_name="customer_segments.csv", mime="text/csv")

# --- Smart Summary ---
st.subheader("🧠 Cluster Summary")
cluster_avg = df.groupby("Cluster")[["Income", "TotalSpend", "Recency", "Age"]].mean().round(2)
for cluster_id, row in cluster_avg.iterrows():
    st.markdown(f"**Cluster {cluster_id}:**")
    st.markdown(f"""
    - 💰 Avg Income: ${row['Income']}
    - 🛍️ Total Spend: {row['TotalSpend']}
    - ⏱️ Recency: {row['Recency']} days
    - 🎂 Avg Age: {row['Age']} years
    """)
