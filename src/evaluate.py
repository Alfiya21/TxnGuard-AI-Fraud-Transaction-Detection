"""
evaluate.py
===========
Model evaluation utilities used by train_model.py and the notebook.

Functions
---------
evaluate_model      — prints classification report + ROC-AUC + PR-AUC
plot_evaluation     — saves ROC curve + Precision-Recall curve + confusion matrix
get_shap_explainer  — returns a cached SHAP TreeExplainer for a fitted model
compute_shap_values — computes real SHAP values for a DataFrame of features
shap_row_to_list    — converts SHAP output → JSON-serialisable list for the API
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# SHAP is an optional import — the API falls back gracefully if missing
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Text evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, label: str = "Model") -> dict:
    """
    Prints classification report, ROC-AUC, and PR-AUC.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, y_pred, y_prob
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    print(f"\n{'='*55}")
    print(f"  {label} — Evaluation Report")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"{'='*55}\n")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "y_pred": y_pred, "y_prob": y_prob}


# ---------------------------------------------------------------------------
# Plot evaluation (ROC + PR + Confusion Matrix)
# ---------------------------------------------------------------------------

def plot_evaluation(
    model,
    X_test,
    y_test,
    label: str = "Model",
    save_path: str = "evaluation.png",
):
    """Saves a 3-panel evaluation figure: ROC curve, PR curve, confusion matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{label} — Model Evaluation", fontsize=14, fontweight="bold")

    # -- ROC curve ----------------------------------------------------------
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name=label)
    axes[0].set_title("ROC Curve")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)

    # -- Precision-Recall curve --------------------------------------------
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=label)
    axes[1].set_title("Precision-Recall Curve")

    # -- Confusion matrix ---------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    im = axes[2].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_xticks([0, 1]); axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Legit", "Fraud"])
    axes[2].set_yticklabels(["Legit", "Fraud"])
    fig.colorbar(im, ax=axes[2])
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Evaluation figure saved → {save_path}")


# ---------------------------------------------------------------------------
# SHAP utilities
# ---------------------------------------------------------------------------

# Module-level cache so the explainer is built once per process
_EXPLAINER_CACHE: dict = {}


def get_shap_explainer(model):
    """
    Returns a cached shap.TreeExplainer for `model`.
    Raises RuntimeError if SHAP is not installed.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError(
            "shap is not installed. Run: pip install shap"
        )
    model_id = id(model)
    if model_id not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[model_id] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[model_id]


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """
    Computes SHAP values for all rows in X.

    Returns
    -------
    shap_values : np.ndarray of shape (n_rows, n_features)
                  — positive = pushes toward fraud, negative = away from fraud
    """
    explainer   = get_shap_explainer(model)
    shap_output = explainer.shap_values(X)

    # XGBoost returns a 2-D array directly.
    # Random Forest returns a list [class0_array, class1_array].
    if isinstance(shap_output, list):
        return shap_output[1]          # class-1 (fraud) SHAP values
    return shap_output


def shap_row_to_list(shap_vals: np.ndarray, X_row: pd.Series) -> list:
    """
    Converts a single row of SHAP values + feature values into a
    JSON-serialisable list suitable for the frontend SHAP bar chart.

    Parameters
    ----------
    shap_vals : 1-D np.ndarray — SHAP values for one row (n_features,)
    X_row     : pd.Series — actual feature values for the same row

    Returns
    -------
    list of dicts: [{feature, shap_value, input_value}, …] sorted by |shap|
    """
    result = []
    for feat, sv, iv in zip(X_row.index, shap_vals, X_row.values):
        result.append({
            "feature":     feat,
            "shap_value":  round(float(sv), 6),
            "input_value": round(float(iv), 4),
        })
    # Sort descending by absolute SHAP value
    result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return result