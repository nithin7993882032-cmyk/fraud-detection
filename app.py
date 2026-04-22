from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables
lr_model = None
rf_model = None
scaler = None
feature_columns = None
model_metrics = {}
sample_df = None


# ─────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────
def generate_sample_dataset(n_samples=1000):
    np.random.seed(42)

    n_legit = int(n_samples * 0.95)
    n_fraud = n_samples - n_legit

    legit = pd.DataFrame({
        "transaction_amount": np.random.lognormal(4.5, 1.2, n_legit),
        "time_since_last_txn": np.random.exponential(24, n_legit),
        "num_transactions_today": np.random.poisson(3, n_legit),
        "distance_from_home": np.random.exponential(15, n_legit),
        "is_foreign_transaction": np.random.choice([0,1], n_legit, p=[0.92,0.08]),
        "card_present": np.random.choice([0,1], n_legit, p=[0.15,0.85]),
        "hour_of_day": np.random.choice(range(6,23), n_legit),
        "merchant_risk_score": np.random.beta(2,8,n_legit),
        "is_fraud": 0
    })

    fraud = pd.DataFrame({
        "transaction_amount": np.random.lognormal(6.0,1.5,n_fraud),
        "time_since_last_txn": np.random.exponential(2,n_fraud),
        "num_transactions_today": np.random.poisson(10,n_fraud),
        "distance_from_home": np.random.exponential(80,n_fraud),
        "is_foreign_transaction": np.random.choice([0,1], n_fraud, p=[0.4,0.6]),
        "card_present": np.random.choice([0,1], n_fraud, p=[0.7,0.3]),
        "hour_of_day": np.random.choice(list(range(0,5))+list(range(22,24)), n_fraud),
        "merchant_risk_score": np.random.beta(6,3,n_fraud),
        "is_fraud": 1
    })

    df = pd.concat([legit, fraud]).sample(frac=1)
    return df


# ─────────────────────────────────────────────────────────────
# BALANCING
# ─────────────────────────────────────────────────────────────
def balance_dataset(X, y):
    df = pd.concat([X, y], axis=1)
    majority = df[df["is_fraud"] == 0]
    minority = df[df["is_fraud"] == 1]

    minority_upsampled = resample(minority, replace=True,
                                 n_samples=len(majority), random_state=42)

    balanced = pd.concat([majority, minority_upsampled])
    return balanced.drop("is_fraud", axis=1), balanced["is_fraud"]


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def train_models(df):
    global lr_model, rf_model, scaler, feature_columns, model_metrics

    feature_columns = [
        "transaction_amount","time_since_last_txn","num_transactions_today",
        "distance_from_home","is_foreign_transaction","card_present",
        "hour_of_day","merchant_risk_score"
    ]

    X = df[feature_columns]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train, y_train = balance_dataset(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=50)
    rf_model.fit(X_train, y_train)

    preds = lr_model.predict(X_test)

    model_metrics = {
        "accuracy": round(accuracy_score(y_test, preds)*100,2),
        "precision": round(precision_score(y_test, preds)*100,2),
        "recall": round(recall_score(y_test, preds)*100,2),
        "f1": round(f1_score(y_test, preds)*100,2)
    }


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    global sample_df

    if sample_df is None:
        sample_df = generate_sample_dataset(1000)
        train_models(sample_df)

    return render_template("index.html", metrics=model_metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([
        float(data.get("transaction_amount",0)),
        float(data.get("time_since_last_txn",0)),
        float(data.get("num_transactions_today",0)),
        float(data.get("distance_from_home",0)),
        float(data.get("is_foreign_transaction",0)),
        float(data.get("card_present",0)),
        float(data.get("hour_of_day",12)),
        float(data.get("merchant_risk_score",0.1))
    ]).reshape(1,-1)

    X_scaled = scaler.transform(features)

    prob = rf_model.predict_proba(X_scaled)[0][1]

    return jsonify({
        "fraud_probability": round(prob*100,2),
        "result": "Fraud" if prob > 0.5 else "Legit"
    })


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)