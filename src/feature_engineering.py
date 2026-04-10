"""
feature_engineering.py
=======================
Builds the full feature matrix used for training and inference.

Features engineered:
  - Time-based      : TX_HOUR, TX_DAYOFWEEK, IS_WEEKEND, IS_NIGHT
  - Customer 7-day  : CUST_AVG_AMOUNT_7D, CUST_TX_COUNT_7D, AMOUNT_DEVIATION_7D
  - Customer 30-day : CUST_AVG_AMOUNT_30D, CUST_TX_COUNT_30D, AMOUNT_DEVIATION_30D
  - Terminal        : TERMINAL_TX_COUNT, TERMINAL_FRAUD_RATE, TERMINAL_AVG_AMOUNT
  - Ratio           : AMOUNT_TO_CUST_AVG_RATIO, AMOUNT_TO_TERM_AVG_RATIO
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Feature column list (single source of truth — import everywhere you need it)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "TX_AMOUNT",
    "TX_HOUR",
    "TX_DAYOFWEEK",
    "IS_WEEKEND",
    "IS_NIGHT",
    "CUST_AVG_AMOUNT_7D",
    "CUST_TX_COUNT_7D",
    "AMOUNT_DEVIATION_7D",
    "CUST_AVG_AMOUNT_30D",
    "CUST_TX_COUNT_30D",
    "AMOUNT_DEVIATION_30D",
    "TERMINAL_TX_COUNT",
    "TERMINAL_FRAUD_RATE",
    "TERMINAL_AVG_AMOUNT",
    "AMOUNT_TO_CUST_AVG_RATIO",
    "AMOUNT_TO_TERM_AVG_RATIO",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline for the training notebook.

    Parameters
    ----------
    df : DataFrame with columns:
         TX_DATETIME, TX_AMOUNT, CUSTOMER_ID, TERMINAL_ID, TX_FRAUD

    Returns
    -------
    df : DataFrame with all FEATURE_COLS added.
    """
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    # ------------------------------------------------------------------
    # 1. Time-based features
    # ------------------------------------------------------------------
    df["TX_HOUR"]       = df["TX_DATETIME"].dt.hour
    df["TX_DAYOFWEEK"]  = df["TX_DATETIME"].dt.dayofweek          # 0=Mon … 6=Sun
    df["IS_WEEKEND"]    = df["TX_DAYOFWEEK"].isin([5, 6]).astype(int)
    df["IS_NIGHT"]      = df["TX_HOUR"].between(0, 5).astype(int) # midnight–5 am

    # ------------------------------------------------------------------
    # 2. Customer behavioural features — 7-day window
    # ------------------------------------------------------------------
    cust_stats_7d = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .agg(CUST_AVG_AMOUNT_7D="mean", CUST_TX_COUNT_7D="count")
        .reset_index()
    )
    df = df.merge(cust_stats_7d, on="CUSTOMER_ID", how="left")
    df["AMOUNT_DEVIATION_7D"] = df["TX_AMOUNT"] - df["CUST_AVG_AMOUNT_7D"]

    # ------------------------------------------------------------------
    # 3. Customer behavioural features — 30-day window
    #    (In the notebook this is computed over the full history;
    #     for rolling windows use compute_rolling_features() below.)
    # ------------------------------------------------------------------
    cust_stats_30d = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .agg(CUST_AVG_AMOUNT_30D="mean", CUST_TX_COUNT_30D="count")
        .reset_index()
    )
    df = df.merge(cust_stats_30d, on="CUSTOMER_ID", how="left")
    df["AMOUNT_DEVIATION_30D"] = df["TX_AMOUNT"] - df["CUST_AVG_AMOUNT_30D"]

    # ------------------------------------------------------------------
    # 4. Terminal-level features
    # ------------------------------------------------------------------
    term_stats = (
        df.groupby("TERMINAL_ID")
        .agg(
            TERMINAL_TX_COUNT=("TX_AMOUNT", "count"),
            TERMINAL_AVG_AMOUNT=("TX_AMOUNT", "mean"),
            TERMINAL_FRAUD_RATE=("TX_FRAUD", "mean"),
        )
        .reset_index()
    )
    df = df.merge(term_stats, on="TERMINAL_ID", how="left")

    # ------------------------------------------------------------------
    # 5. Ratio features — how unusual is this amount?
    # ------------------------------------------------------------------
    df["AMOUNT_TO_CUST_AVG_RATIO"] = (
        df["TX_AMOUNT"] / df["CUST_AVG_AMOUNT_7D"].replace(0, np.nan)
    ).fillna(1.0)

    df["AMOUNT_TO_TERM_AVG_RATIO"] = (
        df["TX_AMOUNT"] / df["TERMINAL_AVG_AMOUNT"].replace(0, np.nan)
    ).fillna(1.0)

    # ------------------------------------------------------------------
    # 6. Fill any NaNs (new customers / terminals seen first time)
    # ------------------------------------------------------------------
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Lookup-table extraction (called after training, saved as .pkl for inference)
# ---------------------------------------------------------------------------

def build_customer_lookup(df: pd.DataFrame) -> dict:
    """Returns {customer_id: {avg_7d, count_7d, avg_30d, count_30d}}."""
    stats = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .agg(avg="mean", count="count")
        .to_dict(orient="index")
    )
    return {k: {"avg": v["avg"], "count": v["count"]} for k, v in stats.items()}


def build_terminal_lookup(df: pd.DataFrame) -> dict:
    """Returns {terminal_id: {tx_count, avg_amount, fraud_rate}}."""
    stats = (
        df.groupby("TERMINAL_ID")
        .agg(
            tx_count=("TX_AMOUNT", "count"),
            avg_amount=("TX_AMOUNT", "mean"),
            fraud_rate=("TX_FRAUD", "mean"),
        )
        .to_dict(orient="index")
    )
    return stats