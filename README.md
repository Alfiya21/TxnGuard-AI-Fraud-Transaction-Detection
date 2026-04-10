# 💳 Fraud Transaction Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9749-brightgreen)

## 🎯 Problem Statement
Financial fraud causes billions in losses annually. This project builds 
an end-to-end ML pipeline to detect fraudulent transactions on a dataset 
of **1.75M+ records** with severe class imbalance (~0.1% fraud rate).

---

## 📊 Dataset
- **Size:** 1.75M+ daily transaction records (`.pkl` format)
- **Features:** Transaction timestamp, Customer ID, Terminal ID, Amount, Fraud label
- **Challenge:** Extreme class imbalance — applied **SMOTE** to improve 
  minority-class representation
- **Fraud scenarios covered:** High-value fraud, terminal compromise, 
  customer credential leakage

---

## ⚙️ Feature Engineering
| Feature Category | Examples |
|---|---|
| Time-based | Hour of transaction, day of week, weekend flag |
| Customer behavior | Avg spend per customer, transaction frequency |
| Terminal risk | Terminal fraud rate, avg terminal transaction amount |

---

## 🤖 Models & Results

| Model | ROC-AUC | Recall | F1-Score |
|---|---|---|---|
| Random Forest | - | - | - |
| **XGBoost** | **0.9749** | **best** | **best** |

- ✅ Reduced false negatives to just **438 fraud cases** via precision 
  threshold optimization
- ✅ Applied **SHAP** for model explainability and transparent fraud flagging

<!-- Add your confusion matrix and ROC curve images here -->
<!-- ![ROC Curve](images/roc_curve.png) -->
<!-- ![Confusion Matrix](images/confusion_matrix.png) -->

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Alfiya21/Fraud-Transaction-Detection-Using-Machine-Learning

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Fraud_Detection.ipynb
```

**Requirements:** Python 3.9+, scikit-learn, xgboost, imbalanced-learn, 
shap, pandas, matplotlib, seaborn

---

## 📁 Project Structure
├── data/                  # Place .pkl dataset files here
├── Fraud_Detection.ipynb  # Main notebook
├── requirements.txt
└── README.md

---

## 👩‍💻 Author
**Alfiya Mulla** — [LinkedIn](www.linkedin.com/in/alfiya-mulla-63893325a) | [GitHub](https://github.com/Alfiya21)
