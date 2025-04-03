import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def segment_customers(df, n_clusters=4):
    """Apply K-Means clustering after scaling features."""
    # Select features for clustering
    features = ["Income", "Recency", "NumWebPurchases"]
    X = df[features]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = model.fit_predict(X_scaled)

    return df, model, scaler

def plot_elbow_method(X_scaled):
    """Plot the elbow method to determine optimal number of clusters."""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.tight_layout()
    plt.show()

def plot_clusters(df):
    """Visualize the clusters using pair plots."""
    sns.pairplot(df, vars=["Income", "Recency", "NumWebPurchases"], hue="Cluster", palette='viridis')
    plt.suptitle('Clusters Visualization', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Dynamically get the correct path to the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "marketing_campaign.csv")

    # Load the dataset with correct separator
    df = pd.read_csv(data_path, sep="\t")

    # Drop rows with missing income values
    df = df.dropna(subset=["Income"])

    # Feature Engineering
    df["Age"] = 2025 - df["Year_Birth"]
    df["TotalSpend"] = df[[
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].sum(axis=1)
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    df["TenureDays"] = (pd.to_datetime("2025-01-01") - df["Dt_Customer"]).dt.days

    # Standardize features and run clustering
    df, model, scaler = segment_customers(df)

    # Plot the elbow method graph
    plot_elbow_method(scaler.transform(df[["Income", "Recency", "NumWebPurchases"]]))

    # Visualize the clusters
    plot_clusters(df)

    # Display preview
    print(df.head())
