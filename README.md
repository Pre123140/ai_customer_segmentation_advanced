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

## ⚖️ How to Run the Project

### 1. 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. ▶️ Launch Streamlit App
```bash
streamlit run app/dashboard.py
```

### 3. 📁 Ensure Dataset Location
The file `marketing_campaign.csv` must be placed in the `/data` directory before launching.

> ⚠️ **Note:** The `customer_segments.csv` file is **not automatically saved to disk**.
> 
> 🟡 It is only offered as a downloadable file from the Streamlit dashboard using:
> ```python
> st.download_button(..., file_name="customer_segments.csv")
> ```
> 
> To save it manually in the code, add:
> ```python
> df.to_csv("outputs/customer_segments.csv", index=False)
> ```

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
🟡 This does not actually save the file to outputs/customer_segments.csv. It only offers a download via Streamlit.
---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Libraries**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `streamlit`
- **Machine Learning**: KMeans, LogisticRegression, PCA

---

## 📚 Documentation
📄 Read the full business & technical walkthrough:  
📘 ['Conceptual Study`](https://github.com/Pre123140/ai_customer_segmentation_advanced/blob/main/AI_POWERED_CUSTOMER_SEGMENTATION%20(ADVANCED).pdf)

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
