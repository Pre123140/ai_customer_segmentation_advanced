# 🧠 AI-Powered Customer Segmentation (Advanced Edition)

A powerful customer segmentation system using **unsupervised and supervised learning** with real retail campaign data. This advanced project blends K-Means clustering, PCA, and logistic regression — delivering explainable insights and interactive Streamlit visualizations for real-world business impact.

---

## 📌 Overview
- 💡 Feature-rich pipeline combining advanced engineering, segmentation, and AI classification.
- 📊 End-to-end analysis: EDA, PCA, clustering, classification, dashboarding.
- 🤖 Logistic Regression to predict high-spending customers.
- 🌐 Deployable Streamlit dashboard with EDA, PCA, clusters, and data export.

---

## 💼 Business Value
- Identify distinct customer segments by behavioral traits and lifecycle stage.
- Predict potential high-value customers for personalized campaigns.
- Improve retention, targeting, and marketing ROI.

---

## 🧠 Core Features
- **Data Preprocessing** — Load, clean, engineer new features like `TotalSpend`, `TenureDays`, `Age`
- **Exploratory Data Analysis (EDA)** — Distribution plots, heatmaps, boxplots by cluster
- **Clustering** — KMeans based on Income, Recency, and Web Purchases
- **Dimensionality Reduction** — PCA for 2D visualization with cluster overlays
- **High-Spender Prediction** — Logistic Regression model with performance metrics
- **Streamlit Dashboard** — Visual interface to view clusters, insights, predictions, and export

---

## 🧪 Dataset
- **File**: `marketing_campaign.csv`
- **Key Features**:
  - Demographics: Age, Marital Status, Education
  - Purchases: MntWines, MntMeatProducts, MntFruits, etc.
  - Engagement: NumWebPurchases, Recency, Dt_Customer

---

## 📂 Folder Structure
```
customer_segmentation_adv/
├── app/
│   └── dashboard.py                     # Streamlit app
├── src/
│   ├── preprocess.py                    # Load & clean data
│   ├── clustering.py                    # KMeans + Elbow + Visuals
│   ├── eda.py                           # Histograms, heatmaps, boxplots
│   ├── ai_insights.py                   # Logistic regression model
│   ├── pca_analysis.py                  # Dashboard logic for PCA/EDA
│   ├── pca_utils.py                     # PCA logic & scatter plots
│   └── __init__.py
├── data/
│   └── marketing_campaign.csv           # Source dataset
├── requirements.txt

```

---

## ▶️ How to Run the Project

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Launch the Streamlit dashboard
streamlit run app/dashboard.py
```

📁 Ensure `data/marketing_campaign.csv` is placed correctly.

---

## 📊 Dashboard Visuals
- PCA scatterplot (PC1 vs PC2) with color-coded clusters
- Cluster distribution countplot
- Boxplots of Income, Spend, and Recency across clusters
- Correlation heatmap across engineered features
- Histograms for Age, Spend, Recency, Tenure

---

## 🤖 AI Logic – High Spender Classifier
- **Definition**: High spenders = customers with `MntWines + MntMeatProducts > 500`
- **Model**: Logistic Regression on features like Income, Recency, Web Purchases
- **Evaluation**: Confusion Matrix and Classification Report
- **Prediction Column**: `SpendingPrediction` added to the final dataset

---

## 📦 Output
- **customer_segments.csv** — Final dataset including:
  - Cluster labels
  - High spender predictions
  - PCA components (PC1, PC2)
  - Demographics and behavioral fields

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Libraries**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `streamlit`
- **Machine Learning**: KMeans, LogisticRegression, PCA

---

## 📚 Documentation
📄 Read the full business & technical walkthrough:  
📘 [`docs/Ai_Customer_Segmentation_Docs.md`](./docs/Ai_Customer_Segmentation_Docs.md)

Includes:
- EDA, PCA, clustering walkthrough
- Visual insights and business value
- Transition from clustering to prediction

---

## 📜 License

This project is open for educational use only. For commercial deployment, contact the author.

---

## 📬 Contact
If you'd like to learn more or collaborate on projects or other initiatives, feel free to connect on [LinkedIn](https://www.linkedin.com/in/prerna-burande-99678a1bb/) or check out my [portfolio site](https://youtheleader.com/).
