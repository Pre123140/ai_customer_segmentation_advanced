import os
import pandas as pd

def load_clean_data():
    # 🔹 Get correct path to dataset
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "marketing_campaign.csv")

    # 🔹 Try comma-separated parsing
    df = pd.read_csv(file_path, sep=",", skipinitialspace=True)

    # 🔹 Show columns for verification
    print("📌 Columns in DataFrame:")
    for col in df.columns:
        print(f"- '{col}'")

    # 🔹 Drop rows with missing Income
    if "Income" in df.columns:
        df.dropna(subset=["Income"], inplace=True)
    else:
        print("❌ 'Income' column still not found. Check the dataset formatting.")
        raise KeyError("Missing 'Income' column.")

    return df

if __name__ == "__main__":
    df = load_clean_data()
    print("✅ Data loaded with shape:", df.shape)
