<div align="center">

  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Telephone%20Receiver.png" alt="Phone" width="80" />

  # Telco Customer Churn Prediction

  **🤖 Machine Learning | 📊 Data Science | 🏆 86% ROC-AUC**

  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
  [![License](https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge)](LICENSE)

  <p align="center">
    <b>Predict customer churn with a tuned Random Forest classifier</b><br>
    <sub>Catch <strong>2 out of 3</strong> actual churners before they leave 🎯</sub>
  </p>

  [🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [🔧 Usage](#-usage-for-inference) • [📁 Structure](#-project-structure)

</div>

---

## 📌 Overview

> **Problem:** Telecom companies lose millions to customer churn.  
> **Solution:** A production-ready ML pipeline that predicts churn probability for each customer.

This project builds an end-to-end **binary classification pipeline** using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset (7,043 customers × 21 features).

### ✨ Highlights

| Feature | Description |
|---------|-------------|
| 🧹 **Auto Cleaning** | Handles missing values, type conversions, and edge cases automatically |
| ⚖️ **SMOTE** | Balances the dataset with Synthetic Minority Over-sampling Technique |
| 🔀 **OHE + Pipeline** | One-Hot Encoding inside a `ColumnTransformer` — no data leakage |
| 🎯 **Tuned RF** | `RandomizedSearchCV` for optimal hyperparameters (ROC-AUC scoring) |
| 💾 **Pickled Model** | Single `pipeline.pkl` file — deploy anywhere |
| 📈 **86% AUC** | Strong discriminative power between churners & loyal customers |

---

## 🗂️ Project Structure

```text
📦 telco-customer-churn/
├── 📁 data/                          # Auto-downloaded dataset
├── 📁 models/                        # 🏆 Saved pipeline (pipeline.pkl)
├── 📁 notebooks/                     # 📓 EDA + training notebook
├── 📁 outputs/                       # 📊 Confusion matrix, ROC curve, etc.
├── 📁 scripts/                       # 🧩 Modular Python modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── 📄 requirements.txt
├── 📄 README.md
└── 🚀 run.py                         # One-command entry point
```

---

## 🚀 Quick Start

### 1️⃣ Clone

```bash
git clone https://github.com/your-username/telco-customer-churn.git
cd telco-customer-churn
```

### 2️⃣ Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install

```bash
pip install -r requirements.txt
```

### 4️⃣ Run

```bash
python run.py
```

> ☕ Grab a coffee. The script will:
> - Download data via `kagglehub`
> - Clean & preprocess
> - Train/test split (80/20)
> - Hyperparameter tuning (10 iter × 5-fold CV)
> - Save the best model to `models/churn_pipeline_tuned.pkl`
> - Export plots to `outputs/`

---

## 📊 Results

### 🏅 Test Set Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | `79.8%` | Overall correct predictions |
| **Precision** | `0.61` | 61% of flagged churners actually churn |
| **Recall** | `0.66` | Catches **66%** of all real churners |
| **F1-Score** | `0.64` | Balanced precision & recall |
| **ROC-AUC** | `0.859` | Excellent class separation |

> 💡 **Business Impact:** Identifies 2 out of 3 churners — perfect for targeted retention campaigns.

### 📈 Visualizations

<div align="center">
  <img src="outputs/confusion_matrix.png" width="45%" alt="Confusion Matrix" />
  &nbsp;
  <img src="outputs/roc_curve.png" width="45%" alt="ROC Curve" />
</div>

---

## 🔧 Usage for Inference

```python
import pickle
import pandas as pd

# 1️⃣ Load the trained pipeline
with open("models/churn_pipeline_tuned.pkl", "rb") as f:
    pipeline = pickle.load(f)

# 2️⃣ Prepare new customer data
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 900.0
}])

# 3️⃣ Predict
prediction = pipeline.predict(new_customer)
probability = pipeline.predict_proba(new_customer)[:, 1]

print(f"🔮 Churn Prediction: {'Yes ⚠️' if prediction[0] == 1 else 'No ✅'}")
print(f"📊 Churn Probability: {probability[0]:.2%}")
```

---

## 🛠️ Customization

| Want to... | Edit this file |
|------------|----------------|
| Change search budget | `n_iter` in `scripts/train.py` |
| Change cross-validation folds | `cv` in `scripts/train.py` |
| Swap model (e.g., XGBoost) | Replace `RandomForestClassifier` + update `param_dist` |
| Add new features | `scripts/preprocessing.py` |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a PR.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your Name](https://linkedin.com/in/your-profile)

---

<div align="center">
  <sub>Built with ❤️ using <a href="https://scikit-learn.org">scikit-learn</a>, <a href="https://imbalanced-learn.org">imbalanced-learn</a>, and <a href="https://pandas.pydata.org">pandas</a></sub><br>
  <sub>Dataset: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">Telco Customer Churn @ Kaggle</a></sub>
</div>
