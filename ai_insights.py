import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict_high_spenders(df):
    """Predict high-spending customers using Logistic Regression (scaled)"""

    df["HighSpender"] = (df["MntWines"] + df["MntMeatProducts"] > 500).astype(int)

    features = ["Income", "Recency", "NumWebPurchases", "TotalSpend", "Age"]
    X = df[features]
    y = df["HighSpender"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic Regression with more iterations
    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y)

    # Prediction
    df["SpendingPrediction"] = model.predict(X_scaled)

    # Evaluation
    print("📊 Classification Report:")
    print(classification_report(y, df["SpendingPrediction"]))

    print("\n🧮 Confusion Matrix:")
    cm = confusion_matrix(y, df["SpendingPrediction"])
    print(cm)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.title("Confusion Matrix - High Spenders")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return df

# -------------------- MAIN SCRIPT --------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "marketing_campaign.csv")

    df = pd.read_csv(data_path, sep="\t")
    df.dropna(subset=["Income"], inplace=True)

    # Feature engineering
    df["Age"] = 2025 - df["Year_Birth"]
    df["TotalSpend"] = df[[
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].sum(axis=1)
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    df["TenureDays"] = (pd.to_datetime("2025-01-01") - df["Dt_Customer"]).dt.days

    # Run prediction logic
    df = predict_high_spenders(df)
