<div align="center">

  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Telephone%20Receiver.png" alt="Phone" width="80" />

  # Telco Customer Churn Prediction

  **🤖 Machine Learning | 📊 Data Science | 🏆 85.82% ROC-AUC**

  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
  [![License](https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge)](LICENSE)

  <p align="center">
    <b>Predict customer churn with an optimized Random Forest pipeline</b><br>
    <sub>Successfully catch vulnerable accounts before they switch operators 🎯</sub>
  </p>

  [🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [🔧 Usage](#-usage-for-inference) • [📁 Structure](#-project-structure)

</div>

---

## 📌 Overview

> **Problem:** Telecom networks lose substantial revenue annually due to sudden subscriber subscription cancellations.  
> **Solution:** An automated, production-ready Machine Learning pipeline that processes raw customer profiles and returns high-accuracy churn risk.

This project implements an end-to-end **binary classification workflow** on the popular Telco Customer Churn dataset, heavily focusing on handling extreme class imbalances cleanly without data leakage.

### ✨ Highlights

| Feature | Description |
|---------|-------------|
| 🧹 **Automated Cleaning** | Handles hidden whitespace missing values in `TotalCharges` safely |
| ⚖️ **Integrated SMOTE** | Combats class imbalance using over-sampling directly inside an `ImbPipeline` |
| 🔀 **Feature Alignment** | Combines `StandardScaler` and `OneHotEncoder` within a unified `ColumnTransformer` |
| 🎯 **Tuned Classifier** | Employs an optimized `RandomForestClassifier` with balanced subsampling weights |
| 💾 **Single Asset Deploy** | Packs everything into a single `churn_model.pkl` — ready for immediate deployment |
| 🌐 **Streamlit App** | Interactive web dashboard for real-time churn prediction |

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/imtiazdeepto/MLProject.git
cd MLProject/Project_1_Customer_churn
```

### 2️⃣ Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn kagglehub streamlit xgboost
```

---

## 📊 Results

### 🏅 Model Performance Evaluation

By isolating `SMOTE` only to our training folds within an `imblearn.pipeline.Pipeline`, the model delivers realistic, high-performing metrics on unseen validation data:

| Metric | Value |
|--------|-------|
| **ROC-AUC Score** | `85.82%` |
| **Accuracy** | `~80%` |
| **Recall (Churn)** | `0.66` |
| **Precision (Churn)** | `0.61` |
| **F1-Score (Churn)** | `0.64` |

> 💡 **Business Impact:** Given the high ROC-AUC and solid probability scoring, retention teams can confidently target vulnerable accounts with proactive offers (e.g., automated loyalty discounts, contract switches) before they terminate service.

### 🔬 Best Hyperparameters Found

```text
rfc__class_weight: balanced_subsample
rfc__max_depth: 10
rfc__max_features: log2
rfc__min_samples_leaf: 5
rfc__min_samples_split: 3
rfc__n_estimators: 439
```

---

## 🔧 Usage for Inference

Because the entire preprocessing stack is bundled into the saved `.pkl` file, you can pass raw data directly into the estimator without transforming the features manually.

```python
import pickle
import pandas as pd

# 1️⃣ Load the trained pipeline
with open("churn_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# 2️⃣ Pass raw customer profile data
new_customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 30,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 2500.0
}])

# 3️⃣ Run Prediction
prediction = pipeline.predict(new_customer)
probability = pipeline.predict_proba(new_customer)[:, 1]

print(f"🔮 Churn Prediction: {'Yes ⚠️' if prediction[0] == 1 else 'No ✅'}")
print(f"📊 Churn Probability: {probability[0]:.2%}")
# Output -> Prediction: 1 (Churn) | Churn Probability: 78.32%
```

---

## 🌐 Streamlit App

Launch the interactive dashboard:

```bash
streamlit run app.py
```

Features:
- 📝 Input customer details via sidebar
- 🔮 Real-time churn prediction
- 📊 Probability gauge visualization

---

## 📁 Project Structure

```text
Project_1_Customer_churn/
├── churn Model.ipynb       # Jupyter Notebook: EDA, training, evaluation
├── churn_model.pkl         # Serialized production pipeline (Preprocessing + RF Model)
├── app.py                  # Streamlit frontend dashboard application
└── README.md               # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR for any pipeline enhancements or deployment wrappers.

---

## 📝 License

Distributed under the **MIT License**. See `LICENSE` for more details.

---

<div align="center">
  <sub>Built with ❤️ using <a href="https://scikit-learn.org">scikit-learn</a>, <a href="https://imbalanced-learn.org">imbalanced-learn</a>, and <a href="https://pandas.pydata.org">pandas</a></sub>
  <br>
  <sub>Dataset Source: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">Telco Customer Churn @ Kaggle</a></sub>
</div>
