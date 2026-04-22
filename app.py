from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

app = Flask(__name__)

rf_model = None
lr_model = None
scaler = None
trained = False

metrics_data = {}


# ---------- DATA ----------
def generate_data(n=1000):
    np.random.seed(42)

    legit = int(n * 0.95)
    fraud = n - legit

    df_legit = pd.DataFrame({
        "amount": np.random.normal(100, 50, legit),
        "distance": np.random.normal(10, 5, legit),
        "transactions": np.random.randint(1, 5, legit),
        "is_fraud": 0
    })

    df_fraud = pd.DataFrame({
        "amount": np.random.normal(1000, 300, fraud),
        "distance": np.random.normal(50, 20, fraud),
        "transactions": np.random.randint(5, 15, fraud),
        "is_fraud": 1
    })

    return pd.concat([df_legit, df_fraud]).sample(frac=1)


# ---------- TRAIN ----------
def train():
    global rf_model, lr_model, scaler, trained, metrics_data

    df = generate_data(1000)

    X = df[["amount", "distance", "transactions"]]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr_model = LogisticRegression()
    rf_model = RandomForestClassifier(n_estimators=50)

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Metrics
    lr_acc = accuracy_score(y_test, lr_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    cm = confusion_matrix(y_test, rf_pred)

    metrics_data = {
        "lr_acc": round(lr_acc * 100, 2),
        "rf_acc": round(rf_acc * 100, 2),
        "cm": cm.tolist(),
        "fraud_count": int(sum(y)),
        "legit_count": int(len(y) - sum(y))
    }

    trained = True


# ---------- ROUTES ----------
@app.route("/")
def home():
    global trained
    if not trained:
        train()
    return render_template("index.html", metrics=metrics_data)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    X = np.array([[ 
        float(data["amount"]),
        float(data["distance"]),
        float(data["transactions"])
    ]])

    X = scaler.transform(X)

    prob = rf_model.predict_proba(X)[0][1]

    if prob > 0.8:
        risk = "HIGH RISK"
    elif prob > 0.5:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return jsonify({
        "probability": round(prob * 100, 2),
        "result": "Fraud" if prob > 0.5 else "Legit",
        "risk": risk
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)