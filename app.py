"""
E-Commerce Fraud Detection Web Application
==========================================
Backend: Flask + Scikit-learn
Author: College Project
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.utils import resample
import json
import io
import base64
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─── Global variables to store trained models ───────────────────────────────
lr_model = None
rf_model = None
scaler = None
feature_columns = None
model_metrics = {}


def generate_sample_dataset(n_samples=5000):
    """
    Generate a realistic dummy fraud dataset.
    In real life, you'd use actual transaction data.
    """
    np.random.seed(42)

    # Legit transactions (majority class ~95%)
    n_legit = int(n_samples * 0.95)
    n_fraud = n_samples - n_legit

    legit = pd.DataFrame({
        "transaction_amount": np.random.lognormal(mean=4.5, sigma=1.2, size=n_legit),
        "time_since_last_txn": np.random.exponential(scale=24, size=n_legit),
        "num_transactions_today": np.random.poisson(lam=3, size=n_legit),
        "distance_from_home": np.random.exponential(scale=15, size=n_legit),
        "is_foreign_transaction": np.random.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        "card_present": np.random.choice([0, 1], size=n_legit, p=[0.15, 0.85]),
        "hour_of_day": np.random.choice(range(6, 23), size=n_legit),
        "merchant_risk_score": np.random.beta(a=2, b=8, size=n_legit),
        "is_fraud": 0
    })

    # Fraud transactions (minority class ~5%)
    fraud = pd.DataFrame({
        "transaction_amount": np.random.lognormal(mean=6.0, sigma=1.5, size=n_fraud),
        "time_since_last_txn": np.random.exponential(scale=2, size=n_fraud),
        "num_transactions_today": np.random.poisson(lam=10, size=n_fraud),
        "distance_from_home": np.random.exponential(scale=80, size=n_fraud),
        "is_foreign_transaction": np.random.choice([0, 1], size=n_fraud, p=[0.4, 0.6]),
        "card_present": np.random.choice([0, 1], size=n_fraud, p=[0.7, 0.3]),
        "hour_of_day": np.random.choice(list(range(0, 5)) + list(range(22, 24)), size=n_fraud),
        "merchant_risk_score": np.random.beta(a=6, b=3, size=n_fraud),
        "is_fraud": 1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    df["transaction_amount"] = df["transaction_amount"].clip(1, 50000).round(2)
    df["time_since_last_txn"] = df["time_since_last_txn"].round(2)
    df["distance_from_home"] = df["distance_from_home"].round(2)
    df["merchant_risk_score"] = df["merchant_risk_score"].round(4)
    return df


def balance_dataset(X, y):
    """
    Handle class imbalance using oversampling (poor man's SMOTE).
    We duplicate minority class samples to balance the dataset.
    """
    df = pd.concat([X, y], axis=1)
    majority = df[df["is_fraud"] == 0]
    minority = df[df["is_fraud"] == 1]

    # Upsample minority class
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )

    balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
    return balanced.drop("is_fraud", axis=1), balanced["is_fraud"]


def train_models(df):
    """
    Train Logistic Regression and Random Forest models.
    Returns metrics for both models.
    """
    global lr_model, rf_model, scaler, feature_columns, model_metrics

    feature_columns = [
        "transaction_amount", "time_since_last_txn", "num_transactions_today",
        "distance_from_home", "is_foreign_transaction", "card_present",
        "hour_of_day", "merchant_risk_score"
    ]

    X = df[feature_columns]
    y = df["is_fraud"]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance training data
    X_train_bal, y_train_bal = balance_dataset(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # ── Logistic Regression ──────────────────────────────────────────
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
    lr_model.fit(X_train_scaled, y_train_bal)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    # ── Random Forest ────────────────────────────────────────────────
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train_bal)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # ── Compute metrics ──────────────────────────────────────────────
    def get_metrics(y_true, y_pred, y_proba, name):
        cm = confusion_matrix(y_true, y_pred)
        return {
            "name": name,
            "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
            "roc_auc": round(roc_auc_score(y_true, y_proba) * 100, 2),
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),
            "fn": int(cm[1][0]), "tp": int(cm[1][1])
        }

    model_metrics = {
        "lr": get_metrics(y_test, lr_pred, lr_proba, "Logistic Regression"),
        "rf": get_metrics(y_test, rf_pred, rf_proba, "Random Forest"),
        "dataset": {
            "total": len(df),
            "fraud": int(y.sum()),
            "legit": int((y == 0).sum()),
            "fraud_pct": round(y.mean() * 100, 2)
        }
    }

    return model_metrics


# ── Auto-train on startup with sample data ──────────────────────────────────
print("🚀 Training models on sample dataset...")
sample_df = generate_sample_dataset(5000)
train_models(sample_df)
print("✅ Models trained successfully!")


# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html", metrics=model_metrics)


@app.route("/api/metrics")
def get_metrics():
    """Return model metrics as JSON for charts."""
    return jsonify(model_metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict fraud for a single transaction.
    Accepts JSON with transaction features.
    """
    try:
        data = request.get_json()
        features = [
            float(data.get("transaction_amount", 0)),
            float(data.get("time_since_last_txn", 0)),
            float(data.get("num_transactions_today", 0)),
            float(data.get("distance_from_home", 0)),
            float(data.get("is_foreign_transaction", 0)),
            float(data.get("card_present", 0)),
            float(data.get("hour_of_day", 12)),
            float(data.get("merchant_risk_score", 0.1))
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        lr_pred = int(lr_model.predict(X_scaled)[0])
        lr_prob = float(lr_model.predict_proba(X_scaled)[0][1])

        rf_pred = int(rf_model.predict(X_scaled)[0])
        rf_prob = float(rf_model.predict_proba(X_scaled)[0][1])

        # Final verdict: average of both models
        avg_prob = (lr_prob + rf_prob) / 2
        final_pred = 1 if avg_prob >= 0.5 else 0

        risk_level = "LOW" if avg_prob < 0.3 else "MEDIUM" if avg_prob < 0.6 else "HIGH"

        return jsonify({
            "success": True,
            "prediction": final_pred,
            "is_fraud": final_pred == 1,
            "fraud_probability": round(avg_prob * 100, 2),
            "legit_probability": round((1 - avg_prob) * 100, 2),
            "risk_level": risk_level,
            "lr_probability": round(lr_prob * 100, 2),
            "rf_probability": round(rf_prob * 100, 2),
            "verdict": "⚠️ FRAUDULENT TRANSACTION" if final_pred == 1 else "✅ LEGITIMATE TRANSACTION"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """
    Upload a CSV file and predict fraud for all rows.
    Returns per-row predictions + summary stats.
    """
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"success": False, "error": "Please upload a CSV file"}), 400

        df = pd.read_csv(file)

        # Check required columns exist
        required = [
            "transaction_amount", "time_since_last_txn", "num_transactions_today",
            "distance_from_home", "is_foreign_transaction", "card_present",
            "hour_of_day", "merchant_risk_score"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing columns: {missing}. Required: {required}"
            }), 400

        # Predict for all rows
        X = df[required].fillna(0)
        X_scaled = scaler.transform(X)

        lr_probs = lr_model.predict_proba(X_scaled)[:, 1]
        rf_probs = rf_model.predict_proba(X_scaled)[:, 1]
        avg_probs = (lr_probs + rf_probs) / 2
        preds = (avg_probs >= 0.5).astype(int)

        df["fraud_probability"] = (avg_probs * 100).round(2)
        df["prediction"] = preds
        df["verdict"] = df["prediction"].map({0: "Legitimate", 1: "Fraud"})

        total = len(df)
        fraud_count = int(preds.sum())
        legit_count = total - fraud_count

        results = df.head(50).to_dict(orient="records")  # First 50 rows

        return jsonify({
            "success": True,
            "total": total,
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_pct": round(fraud_count / total * 100, 2),
            "results": results
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Retrain on a fresh sample dataset."""
    global sample_df
    sample_df = generate_sample_dataset(5000)
    metrics = train_models(sample_df)
    return jsonify({"success": True, "metrics": metrics})


@app.route("/api/sample-csv")
def download_sample():
    """Generate and return a sample CSV for testing uploads."""
    df = generate_sample_dataset(100)
    df = df.drop("is_fraud", axis=1)  # Don't give away labels!
    csv_str = df.to_csv(index=False)
    return app.response_class(
        csv_str,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"}
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
