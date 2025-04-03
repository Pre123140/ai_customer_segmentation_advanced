# 🧠 AI-Powered Customer Segmentation (Advanced Edition)

A robust, machine learning-powered customer segmentation project built using Python, EDA, PCA, and K-Means clustering. This advanced version enables smart segmentation, spending prediction, and visual interpretation — all delivered through an interactive Streamlit dashboard.

---

## 📌 Overview

- 🎯 Combines feature engineering, lifecycle analysis, and PCA for enhanced segmentation.
- 📊 Uses exploratory data analysis (EDA) to derive actionable customer insights.
- 🤖 Incorporates a logistic regression model to predict high-spending customers.
- 🌐 Interactive Streamlit dashboard with visualizations, cluster analytics, and CSV export.

---

## 📁 Folder Structure
customer_segmentation_adv/
├── app/
│   └── dashboard.py                  # Streamlit dashboard interface
│
├── src/
│   ├── preprocess.py                # Data cleaning & feature engineering
│   ├── clustering.py                # KMeans clustering logic
│   ├── pca_analysis.py              # PCA transformation and plotting
│   ├── eda.py                       # EDA visualizations and summaries
│   ├── ai_insights.py               # Classification model for high spenders
│   └── __init__.py
│
├── data/
│   └── marketing_campaign.csv       # Raw input dataset
│
├── outputs/
│   └── customer_segments.csv        # Exportable segmented data (from dashboard)
│
├── docs/
│   └── Ai_Customer_Segmentation_Docs.md  # Detailed documentation & case study
│
├── README.md                        # Project summary and instructions
└── requirements.txt                 # Required Python libraries



---

## 🚀 How to Run the Dashboard

1. 📦 Install dependencies

```bash
pip install -r requirements.txt


## 🚀 How to Run the Dashboard

1. 📦 Install dependencies

```bash
pip install -r requirements.txt

▶️ Launch the Streamlit dashboard
streamlit run app/dashboard.py

📁 Ensure the dataset is located at:
data/marketing_campaign.csv

🧠 Features
Feature	                                                          Description
Feature Engineering                                              	Derived columns: Age, TotalSpend, TenureDays, TotalChildren
PCA	                                                              Dimensionality reduction + 2D cluster visualization
EDA	                                                              Histograms, boxplots, correlation heatmaps
KMeans Clustering	                                                Segmentation based on behavior, spend, and engagement
AI-Powered Predictions                                           	Predicts high spenders using logistic regression
Interactive Dashboard                                             	View insights and export results with Streamlit


📊 Key Visualizations (In-Dashboard)
PCA Scatter Plot (with cluster coloring)

Cluster Distribution Countplot

Boxplots (Income, Recency, TotalSpend per cluster)

Correlation Heatmap

Histograms (Age, Tenure, Spend)



🛠️ Tech Stack
Languages: Python 3.9+

Libraries: pandas, seaborn, matplotlib, scikit-learn, Streamlit

Machine Learning: PCA, KMeans, LogisticRegression

Visualization: Seaborn, Matplotlib, Streamlit Components



📚 Documentation
Complete case study, conceptual explanation, and business impact can be found in:

docs/ai_powered_customer_segmentation_advanced.md

Includes:
Business value of segmentation
EDA and PCA use cases
Unsupervised → semi-supervised AI transition
Future of segmentation in 2025 and beyond



📥 Dataset
marketing_campaign.csv used as the raw dataset (source: public retail dataset)

Must be located in the /data folder before launching the app


📦 Output
customer_segments.csv generated via dashboard after clustering & prediction

Includes cluster labels and high spender predictions


🙌 Author
👩‍💻 Prerna Burande
📌 Portfolio: 
🔗 LinkedIn: 

📝 License
This project is licensed under the MIT License.
