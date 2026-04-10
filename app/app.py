"""
app.py — TxnGuard Flask application
Routes:
    GET  /          → index.html
    POST /predict   → JSON prediction + real SHAP
    GET  /health    → Render liveness probe
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from predictor import predict_from_api

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "templates"),
)

MODEL_STATS = {
    "roc_auc":            0.9749,
    "precision":          0.94,
    "recall":             0.91,
    "f1":                 0.92,
    "false_negatives":    438,
    "total_transactions": "1.75M+",
}


@app.route("/")
def home():
    return render_template("index.html", stats=MODEL_STATS)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or empty JSON body"}), 400
    try:
        result = predict_from_api(data)
        return jsonify(result)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)