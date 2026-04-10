"""
train_model.py
==============
Training pipeline for the fraud detection models.

Steps
-----
1. Load all daily .pkl files from dataset/data/
2. Apply full feature engineering
3. Handle class imbalance with SMOTE
4. Train XGBoost (primary) and Random Forest (baseline)
5. Save models + lookup tables to models/

Run from project root:
    python src/train_model.py
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from feature_engineering import (
    create_features,
    build_customer_lookup,
    build_terminal_lookup,
    FEATURE_COLS,
)
from evaluate import evaluate_model, plot_evaluation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "dataset", "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_all_transactions(data_dir: str) -> pd.DataFrame:
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    frames = [pd.read_pickle(f) for f in pkl_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"[DATA]  Loaded {len(pkl_files)} files → {len(df):,} transactions")
    print(f"[DATA]  Fraud rate: {df['TX_FRAUD'].mean():.4%}")
    return df


# ---------------------------------------------------------------------------
# 2. Train XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# 3. Train Random Forest (baseline)
# ---------------------------------------------------------------------------
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 4. Main training pipeline
# ---------------------------------------------------------------------------
def main():
    # ---- Load & engineer features ----------------------------------------
    df_raw = load_all_transactions(DATA_DIR)
    df     = create_features(df_raw)

    X = df[FEATURE_COLS]
    y = df["TX_FRAUD"]

    # ---- Train / test split (time-ordered: last 20% as test) --------------
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"\n[SPLIT] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"[SPLIT] Train fraud rate: {y_train.mean():.4%}")
    print(f"[SPLIT] Test  fraud rate: {y_test.mean():.4%}")

    # ---- SMOTE on training set only (never on test!) ----------------------
    print("\n[SMOTE] Applying SMOTE oversampling …")
    smote = SMOTE(sampling_strategy=0.1, random_state=42, n_jobs=-1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] After resampling: {len(X_train_res):,} rows  "
          f"| Fraud rate: {y_train_res.mean():.4%}")

    # ---- Train XGBoost ----------------------------------------------------
    print("\n[XGB]   Training XGBoost …")
    xgb_model = train_xgboost(X_train_res, y_train_res)
    print("[XGB]   Evaluating …")
    evaluate_model(xgb_model, X_test, y_test, label="XGBoost")
    plot_evaluation(xgb_model, X_test, y_test, label="XGBoost",
                    save_path=os.path.join(MODEL_DIR, "xgb_evaluation.png"))

    # ---- Train Random Forest ----------------------------------------------
    print("\n[RF]    Training Random Forest …")
    rf_model = train_random_forest(X_train_res, y_train_res)
    print("[RF]    Evaluating …")
    evaluate_model(rf_model, X_test, y_test, label="Random Forest")

    # ---- Save primary model -----------------------------------------------
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    print("\n[SAVE]  Saved fraud_model.pkl (XGBoost)")

    # ---- Build and save lookup tables ------------------------------------
    cust_stats    = build_customer_lookup(df_raw)
    term_stats    = build_terminal_lookup(df_raw)

    # Split for backward compatibility with predictor.py
    customer_avg   = {k: v["avg"]   for k, v in cust_stats.items()}
    customer_count = {k: v["count"] for k, v in cust_stats.items()}
    terminal_count = {k: v["tx_count"]    for k, v in term_stats.items()}
    terminal_fraud = {k: v["fraud_rate"]  for k, v in term_stats.items()}
    terminal_avg   = {k: v["avg_amount"]  for k, v in term_stats.items()}

    joblib.dump(customer_avg,   os.path.join(MODEL_DIR, "customer_avg.pkl"))
    joblib.dump(customer_count, os.path.join(MODEL_DIR, "customer_count.pkl"))
    joblib.dump(terminal_count, os.path.join(MODEL_DIR, "terminal_count.pkl"))
    joblib.dump(terminal_fraud, os.path.join(MODEL_DIR, "terminal_fraud.pkl"))
    joblib.dump(terminal_avg,   os.path.join(MODEL_DIR, "terminal_avg.pkl"))

    print("[SAVE]  Lookup tables saved (customer_avg, customer_count, "
          "terminal_count, terminal_fraud, terminal_avg)")
    print("\n✅  Training complete.")


if __name__ == "__main__":
    main()