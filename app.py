from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

lr_model = None
rf_model = None
scaler = None
trained = False


# -------- DATA --------
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


# -------- TRAIN --------
def train():
    global lr_model, rf_model, scaler, trained

    df = generate_data(1000)

    X = df[["amount", "distance", "transactions"]]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    lr_model = LogisticRegression()
    rf_model = RandomForestClassifier(n_estimators=50)

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    trained = True


# -------- ROUTES --------
@app.route("/")
def home():
    global trained
    if not trained:
        train()
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    amount = float(data.get("amount", 0))
    distance = float(data.get("distance", 0))
    transactions = float(data.get("transactions", 0))

    X = np.array([[amount, distance, transactions]])
    X = scaler.transform(X)

    prob = rf_model.predict_proba(X)[0][1]

    return jsonify({
        "probability": round(prob * 100, 2),
        "result": "Fraud" if prob > 0.5 else "Legit"
    })


# -------- MAIN --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)