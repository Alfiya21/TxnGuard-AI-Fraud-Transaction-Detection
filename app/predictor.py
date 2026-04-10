"""
predictor.py — TxnGuard
Builds exactly the 9 features the model was trained on.
No extra columns, no renaming, no leakage from other sources.

Training features (from Fraud_Detection.ipynb, cell 8):
    TX_AMOUNT, TX_HOUR, TX_DAYOFWEEK, IS_WEEKEND,
    CUST_AVG_AMOUNT, CUST_TX_COUNT, AMOUNT_DEVIATION,
    TERMINAL_TX_COUNT, TERMINAL_FRAUD_RATE
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# ── load artefacts ────────────────────────────────────────────────────────────
model          = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
customer_avg   = joblib.load(os.path.join(MODELS_DIR, "customer_avg.pkl"))
customer_count = joblib.load(os.path.join(MODELS_DIR, "customer_count.pkl"))
terminal_count = joblib.load(os.path.join(MODELS_DIR, "terminal_count.pkl"))
terminal_fraud = joblib.load(os.path.join(MODELS_DIR, "terminal_fraud.pkl"))

# TreeExplainer — built once at startup, fast for RF & XGBoost
explainer = shap.TreeExplainer(model)

# ── EXACT feature list the saved model was trained on ─────────────────────────
# DO NOT add, remove, or rename any column here.
FEATURES = [
    "TX_AMOUNT",
    "TX_HOUR",
    "TX_DAYOFWEEK",
    "IS_WEEKEND",
    "CUST_AVG_AMOUNT",
    "CUST_TX_COUNT",
    "AMOUNT_DEVIATION",
    "TERMINAL_TX_COUNT",
    "TERMINAL_FRAUD_RATE",
]

FEATURE_LABELS = {
    "TX_AMOUNT":           "Transaction amount",
    "TX_HOUR":             "Hour of transaction",
    "TX_DAYOFWEEK":        "Day of week",
    "IS_WEEKEND":          "Weekend flag",
    "CUST_AVG_AMOUNT":     "Customer avg spend",
    "CUST_TX_COUNT":       "Customer tx count",
    "AMOUNT_DEVIATION":    "Amount vs customer avg",
    "TERMINAL_TX_COUNT":   "Terminal tx volume",
    "TERMINAL_FRAUD_RATE": "Terminal fraud rate",
}


# ── safe lookup helpers ───────────────────────────────────────────────────────
# The saved .pkl files are pandas Series (index = customer/terminal ID).
# A plain series[key] can return an array if the index has duplicates.
# _lookup() always returns a single Python scalar safely.

def _lookup(series, key, default=0.0):
    """Safely get a scalar value from a pandas Series or dict."""
    try:
        if key is None:
            return default
        val = series[key]
        # If it's array-like (duplicate index), take first element
        if hasattr(val, "__len__") and not isinstance(val, str):
            val = val.iloc[0] if hasattr(val, "iloc") else val[0]
        return float(val)
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


# ── feature builder ───────────────────────────────────────────────────────────

def build_features(data: dict) -> pd.DataFrame:
    """
    Maps frontend JSON → exact 9 model features.

    Frontend payload keys:
        tx_amount, tx_hour, tx_day_of_week, is_weekend,
        customer_avg_amount_7d, customer_tx_count_7d,
        customer_tx_count_30d, terminal_fraud_rate
    Optional:
        customer_id, terminal_id  (uses lookup Series if present)
    """

    # transaction basics
    tx_amount  = _safe_float(data.get("tx_amount", 0))
    tx_hour    = _safe_int(data.get("tx_hour", 0))
    tx_dow     = _safe_int(data.get("tx_day_of_week", 0))
    is_weekend = _safe_int(data.get("is_weekend", 0))

    # customer features — lookup by ID if provided, else use form values
    cust_id = data.get("customer_id")
    try:
        cust_id_key = int(float(cust_id)) if cust_id is not None else None
    except (TypeError, ValueError):
        cust_id_key = None

    if cust_id_key is not None:
        cust_avg = _lookup(customer_avg,   cust_id_key, default=0.0)
        cust_cnt = int(_lookup(customer_count, cust_id_key, default=0))
    else:
        cust_avg = _safe_float(data.get("customer_avg_amount_7d", 0))
        cust_cnt = _safe_int(data.get("customer_tx_count_7d", 0))

    # AMOUNT_DEVIATION is always computed — never accepted from payload
    amount_deviation = tx_amount - cust_avg

    # terminal features — lookup by ID if provided, else use form values
    term_id = data.get("terminal_id")
    try:
        term_id_key = int(float(term_id)) if term_id is not None else None
    except (TypeError, ValueError):
        term_id_key = None

    if term_id_key is not None:
        term_cnt   = int(_lookup(terminal_count, term_id_key, default=0))
        term_fraud = _lookup(terminal_fraud, term_id_key, default=0.0)
    else:
        term_cnt   = _safe_int(data.get("customer_tx_count_30d", 0))
        term_fraud = _safe_float(data.get("terminal_fraud_rate", 0.0))

    # assemble in EXACT training order
    row = {
        "TX_AMOUNT":           tx_amount,
        "TX_HOUR":             tx_hour,
        "TX_DAYOFWEEK":        tx_dow,
        "IS_WEEKEND":          is_weekend,
        "CUST_AVG_AMOUNT":     cust_avg,
        "CUST_TX_COUNT":       cust_cnt,
        "AMOUNT_DEVIATION":    amount_deviation,
        "TERMINAL_TX_COUNT":   term_cnt,
        "TERMINAL_FRAUD_RATE": term_fraud,
    }

    return pd.DataFrame([row], columns=FEATURES)


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(df: pd.DataFrame) -> list:
    """Real SHAP via TreeExplainer. Returns list sorted by |shap_value| desc."""
    raw = explainer.shap_values(df[FEATURES])

    # RandomForest binary → list [class0_array, class1_array]
    # XGBoost binary      → 2-D ndarray (n_rows, n_features)
    if isinstance(raw, list):
        # Extract fraud class SHAP values for first row
        sv_array = raw[1][0]  # shape: (n_features,)
    else:
        # Extract first row
        sv_array = raw[0]     # shape: (n_features,)

    # Ensure we have a 1D array of scalars
    sv = np.asarray(sv_array, dtype=float).flatten()

    out = []
    for feat, val in zip(FEATURES, sv):
        # Convert numpy scalar to Python float safely
        shap_val = float(val)
        input_val = float(df[feat].iloc[0])
        
        out.append({
            "feature":     feat,
            "label":       FEATURE_LABELS[feat],
            "shap_value":  round(shap_val, 5),
            "input_value": round(input_val, 4),
            "direction":   "up" if shap_val >= 0 else "down",
        })

    out.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return out


# ── risk helpers ──────────────────────────────────────────────────────────────

def _risk_tier(prob: float) -> str:
    if prob >= 0.80: return "critical"
    if prob >= 0.50: return "high"
    if prob >= 0.25: return "medium"
    return "low"


def _rule_reasons(row: pd.Series) -> list:
    reasons = []
    if row["TX_AMOUNT"] > 500:
        reasons.append("Unusually high transaction amount")
    if row["CUST_AVG_AMOUNT"] > 0 and row["AMOUNT_DEVIATION"] > row["CUST_AVG_AMOUNT"] * 1.5:
        reasons.append("Amount significantly above customer's usual spending")
    if row["TX_HOUR"] < 5 or row["TX_HOUR"] > 22:
        reasons.append("Transaction at unusual hour (late night / early morning)")
    if row["TERMINAL_FRAUD_RATE"] > 0.05:
        reasons.append(f"Terminal has elevated fraud history ({row['TERMINAL_FRAUD_RATE']:.1%})")
    if row["CUST_TX_COUNT"] == 0:
        reasons.append("New or unseen customer — no historical baseline available")
    if not reasons:
        reasons.append("No single dominant flag — model detected suspicious feature combination")
    return reasons


# ── main entry point ──────────────────────────────────────────────────────────

def predict_from_api(data: dict) -> dict:
    """Called by app.py /predict route. Returns JSON-serialisable dict."""
    df   = build_features(data)
    X    = df[FEATURES]                                     # explicit column order
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return {
        "prediction":    pred,
        "label":         "Fraud" if pred == 1 else "Legitimate",
        "probability":   round(prob * 100, 2),
        "risk_tier":     _risk_tier(prob),
        "shap_values":   compute_shap(df),
        "reasons":       _rule_reasons(df.iloc[0]),
        "features_used": X.iloc[0].to_dict(),
    }
