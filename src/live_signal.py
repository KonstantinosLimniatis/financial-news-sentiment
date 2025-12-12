import argparse
import joblib
import os
import pandas as pd

from src.live_features import build_live_features

MODEL_PATH = "models/logistic_live.pkl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    features_file = f"data/processed/{ticker}/news_scored.csv"

    if not os.path.exists(MODEL_PATH):
        print("❌ Live model not found")
        return

    if not os.path.exists(features_file):
        print("⚠️ No live data yet — NO TRADE")
        return

    X_live = build_live_features(features_file)

    if X_live is None or X_live.empty or X_live.isnull().any().any():
        print("⚠️ Insufficient live features — NO TRADE")
        return

    model = joblib.load(MODEL_PATH)
    prob_up = model.predict_proba(X_live)[0][1]

    if prob_up >= 0.65:
        signal = "BUY"
        confidence = "HIGH"
    elif prob_up >= 0.55:
        signal = "BUY"
        confidence = "MEDIUM"
    else:
        signal = "NO TRADE"
        confidence = "LOW"

    print("\n📈 LIVE SIGNAL")
    print(f"Ticker:     {ticker}")
    print(f"Prob(up):   {prob_up:.2f}")
    print(f"Signal:     {signal}")
    print(f"Confidence: {confidence}")

if __name__ == "__main__":
    main()

