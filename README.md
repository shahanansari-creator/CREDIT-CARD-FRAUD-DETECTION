# CREDIT-CARD-FRAUD-DETECTION

# 💳 Credit Card Fraud Detection — End-to-End Data Science Project

An end-to-end exploratory analysis, clustering, and machine learning project built to detect fraudulent credit card transactions in a simulated 100,000-record dataset. The project covers the full data science lifecycle — data cleaning, EDA, feature engineering, unsupervised pattern recognition, supervised classification, and rigorous, imbalance-aware model evaluation.

> 📓 Built to run directly in **Google Colab** — no local setup required.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Report](#-report)
- [License](#-license)

---

## 🔍 Overview

Credit card fraud poses a growing threat to digital commerce as transaction volumes scale. This project analyzes a simulated transaction dataset to:

- Explore transaction patterns across amount, type, location, and time
- Apply clustering to detect natural, unusual transaction groupings
- Build and compare multiple supervised ML models to classify transactions as fraudulent or legitimate
- Evaluate models using metrics suited to **highly imbalanced** classification problems (precision, recall, F1, ROC-AUC) rather than misleading raw accuracy

The project is intentionally transparent about its results — including where the models **did not** perform well — and documents *why*, which is often more valuable than reporting an inflated headline number.

---

## 📊 Dataset

**100,000 simulated credit card transactions** with the following attributes:

| Column | Description |
|---|---|
| `TransactionID` | Unique identifier for each transaction |
| `TransactionDate` | Date & time of the transaction |
| `Amount` | Monetary value of the transaction |
| `MerchantID` | Identifier for the merchant involved |
| `TransactionType` | `purchase` or `refund` |
| `Location` | City where the transaction occurred (10 U.S. cities) |
| `IsFraud` | Target label — `1` = fraud, `0` = legitimate |

**Class distribution:**

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 99,000 | 99.00% |
| Fraud (1) | 1,000 | 1.00% |

No missing values were found across any of the 100,000 rows.

---

## 🗂 Project Structure

```
├── Credit_Card_Fraud_Detection.ipynb   # Full Colab-ready notebook (EDA → Modeling → Evaluation)
├── Credit_Card_Fraud_Detection_Report.docx  # Formal written project report
├── credit_card_fraud_dataset.csv       # Source dataset (100K transactions)
└── README.md
```

---

## 🧠 Methodology

### 1. Data Preparation
- Parsed `TransactionDate` into datetime; derived `Hour` and `DayOfWeek` features
- Label-encoded `TransactionType` and `Location`
- Verified data completeness (no nulls, no duplicates)

### 2. Exploratory Data Analysis (EDA)
- Distribution of transaction amounts (overall + fraud vs. legitimate)
- Fraud rate by transaction type, location, and hour of day
- Correlation heatmap across numeric features

### 3. Pattern Recognition (Unsupervised)
- **K-Means clustering** (k=4) on standardized features
- **PCA** for 2D visualization of transaction clusters
- Fraud-rate comparison across clusters to detect anomalous groupings

### 4. Predictive Modeling (Supervised)
Trained and compared **3 classification models** on an 80/20 stratified train-test split:

| Model | Configuration |
|---|---|
| Logistic Regression | `class_weight='balanced'`, standardized features |
| Decision Tree | `class_weight='balanced'`, `max_depth=8` |
| Random Forest | `class_weight='balanced'`, `n_estimators=200`, `max_depth=10` |

### 5. Evaluation
Since fraud makes up only ~1% of the data, **accuracy alone is misleading**. Models were evaluated using:
- **Precision** — how many flagged transactions were actually fraud
- **Recall** — how many actual fraud cases were caught
- **F1-score** — balance of precision & recall
- **ROC-AUC** — overall separability between classes
- Confusion matrices & ROC curves for threshold-level comparison

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 50.82% | 0.80% | 39.00% | 0.0156 | 0.4564 |
| Decision Tree | 59.72% | 0.97% | 39.00% | 0.0190 | 0.4779 |
| Random Forest | 94.73% | 0.46% | 2.00% | 0.0075 | 0.4750 |

**Feature importance (Random Forest):**

| Feature | Importance |
|---|---|
| Amount | 0.4506 |
| Hour | 0.2304 |
| Location (encoded) | 0.1505 |
| Day of Week | 0.1304 |
| Transaction Type (encoded) | 0.0380 |

---

## 💡 Key Insights

- All three models produced **ROC-AUC scores near 0.50** — equivalent to random guessing — despite Random Forest's misleadingly high 94.73% accuracy (driven by predicting the majority class, catching only 2% of real fraud).
- Fraud rate was nearly identical across transaction types (0.99% purchases vs. 1.01% refunds) and varied only marginally by location (0.89%–1.16%) and hour of day.
- Clustering found **no natural segment** with an elevated concentration of fraud (all 4 clusters had fraud rates within 0.89%–1.09% of the 1% baseline).
- **Conclusion:** the available features (amount, type, location, time) carry little to no statistical relationship with the fraud label in this dataset — most consistent with fraud being randomly assigned during data simulation, rather than being tied to realistic behavioral fraud indicators.
- This is a deliberately honest result: rather than overselling a misleading accuracy score, the project documents *why* the models underperform and what data would be needed to fix it — a critical skill in real-world data science and risk analytics.

---

## 🛠 Tech Stack

- **Python 3** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Machine Learning** — scikit-learn (Logistic Regression, Decision Tree, Random Forest, K-Means, PCA)
- **Environment** — Google Colab

---

## ▶️ How to Run

1. Open `Credit_Card_Fraud_Detection.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run all cells from top to bottom (`Shift + Enter`)
3. When prompted, upload `credit_card_fraud_dataset.csv`
4. All EDA charts, clustering results, and model evaluation metrics will render inline

No installation required — all libraries used are pre-installed on Colab.

---

## ⚠️ Limitations

- Dataset appears synthetically generated with fraud labels largely independent of the provided features
- Only 5 basic features available; no cardholder-level history, device/IP data, or merchant risk scoring
- `MerchantID` excluded from modeling due to high cardinality without merchant-level aggregation
- No ground-truth validation against real-world fraud patterns

---

## 🔭 Future Work

- Engineer cardholder-level features (spending baseline, transaction velocity, deviation from typical behavior)
- Apply **SMOTE** or targeted undersampling to further address class imbalance
- Experiment with **XGBoost / LightGBM** and anomaly-detection methods (Isolation Forest, Autoencoders)
- Tune classification thresholds based on business cost trade-offs (missed fraud vs. false alarms)
- Validate on a richer, more realistic labeled dataset before any production use

---

## 📄 Report

A full written project report (`Credit_Card_Fraud_Detection_Report.docx`) is included, covering:
Executive Summary, Methodology, Results with visualizations, Discussion, Limitations, and Recommendations — written for both technical and non-technical stakeholders.

---

## 📝 License

This project is for educational and portfolio purposes. Feel free to fork, adapt, and build on it.
