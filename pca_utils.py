# src/pca_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_pca(df, features, n_components=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(scaled_data)

    df_copy = df.copy().reset_index(drop=True)
    pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(n_components)])
    df_combined = pd.concat([df_copy, pca_df], axis=1)

    return df_combined, pca

def plot_pca_variance(pca):
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by PCA Components")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pca_clusters(df):
    if "PC1" not in df.columns or "PC2" not in df.columns or "Cluster" not in df.columns:
        print("⚠️ Required columns not found. Cannot plot PCA clusters.")
        return

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="viridis", data=df, alpha=0.7)
    plt.title("Customer Segments Visualized with PCA")
    plt.tight_layout()
    plt.show()
