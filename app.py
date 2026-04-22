from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Simple dummy prediction (no sklearn issues)
def predict_fraud(amount, distance, transactions):
    score = (amount * 0.002) + (distance * 0.01) + (transactions * 0.1)
    return min(score, 1.0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    amount = float(data.get("amount", 0))
    distance = float(data.get("distance", 0))
    transactions = float(data.get("transactions", 0))

    prob = predict_fraud(amount, distance, transactions)

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