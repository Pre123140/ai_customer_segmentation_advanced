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


---

## 🚀 How to Run the Dashboard

1. 📦 Install dependencies

```bash
pip install -r requirements.txt




---

## 🚀 How to Run the Dashboard

1. 📦 Install dependencies

```bash
pip install -r requirements.txt

▶️ Launch the Streamlit dashboard

bash
Copy
Edit
streamlit run app/dashboard.py
📁 Ensure the dataset is located at:

bash
Copy
Edit
data/marketing_campaign.csv
🧠 Features
Feature	Description
Feature Engineering	Derived columns: Age, TotalSpend, TenureDays, TotalChildren
PCA	Dimensionality reduction + 2D cluster visualization
EDA	Histograms, boxplots, correlation heatmaps
KMeans Clustering	Segmentation based on behavior, spend, and engagement
AI-Powered Predictions	Predicts high spenders using logistic regression
Interactive Dashboard	View insights and export results with Streamlit
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

Copy
Edit
docs/Ai_Customer_Segmentation_Docs.md
Includes:

Business value of segmentation

EDA and PCA use cases

Unsupervised → semi-supervised AI transition

Future of segmentation in 2025 and beyond

📷 Screenshots (Optional)
You can add visual previews of your dashboard by uploading images to /assets/ or linking them here.

📥 Dataset
marketing_campaign.csv used as the raw dataset (source: public retail dataset)

Must be located in the /data folder before launching the app

📦 Output
customer_segments.csv generated via dashboard after clustering & prediction

Includes cluster labels and high spender predictions

✅ GitHub Upload Checklist
 app/dashboard.py

 All source scripts in src/

 data/marketing_campaign.csv

 outputs/customer_segments.csv

 docs/Ai_Customer_Segmentation_Docs.md

 README.md

 requirements.txt

🙌 Author
👩‍💻 Prerna Burande
📌 Portfolio: 
🔗 LinkedIn: 

📝 License
This project is licensed under the MIT License.

🏷 Tags
AI Customer Segmentation EDA PCA Clustering Streamlit Dashboard Unsupervised Learning 2025 Business Strategy Retail Analytics

yaml
Copy
Edit

---

Would you like me to now:
- Generate the `requirements.txt` file for this project?
- Create placeholder dashboard screenshots for GitHub?
- Or help you push it all to your GitHub repo with instructions?
