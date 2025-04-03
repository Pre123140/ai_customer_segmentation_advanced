import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional for environments where plots don’t display
# import matplotlib
# matplotlib.use("TkAgg")  # or "QtAgg" depending on OS

def perform_eda(df):
    # -------------------- BASIC INFO --------------------
    print("\U0001F4C4 Basic Data Info:")
    print(df.info())

    print("\n\U0001F4CA Summary Statistics:")
    print(df.describe())

    # -------------------- VALUE COUNTS --------------------
    print("\n\U0001F4CC Categorical Value Distributions:")
    for col in ["Education", "Marital_Status"]:
        print(f"\n{col}:\n", df[col].value_counts())

    # -------------------- DISTRIBUTIONS --------------------
    numeric_cols = ["Age", "Income", "TotalSpend", "Recency", "TenureDays"]
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # -------------------- CORRELATION HEATMAP --------------------
    plt.figure(figsize=(12, 8))
    corr = df[[
        "Age", "Income", "Recency", "TenureDays",
        "TotalSpend", "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"
    ]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # -------------------- BOXPLOTS --------------------
    for col in ["Income", "TotalSpend", "Recency"]:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color="orange")
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

    # -------------------- CLUSTER INSIGHTS (OPTIONAL) --------------------
    if "Cluster" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Cluster", y="TotalSpend", data=df)
        plt.title("Total Spend per Cluster")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Cluster", y="Income", data=df)
        plt.title("Income per Cluster")
        plt.tight_layout()
        plt.show()


# -------------------- STANDALONE EXECUTION --------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "marketing_campaign.csv")

    df = pd.read_csv(data_path, sep="\t")
    df.dropna(subset=["Income"], inplace=True)

    # Feature Engineering
    df["Age"] = 2025 - df["Year_Birth"]
    df["TotalSpend"] = df[[
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].sum(axis=1)
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    df["TenureDays"] = (pd.to_datetime("2025-01-01") - df["Dt_Customer"]).dt.days

    # Run EDA
    perform_eda(df)
