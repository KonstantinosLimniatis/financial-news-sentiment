import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

DATA_FILE = "data/processed/events.csv"
MODEL_OUT = "models/logistic_live.pkl"

FEATURES = [
    "sentiment_score",
    "avg_confidence",
    "news_count",
]

def main():
    df = pd.read_csv(DATA_FILE)

    if df.empty or df["label_up"].nunique() < 2:
        print("❌ Not enough data")
        return

    X = df[FEATURES]
    y = df["label_up"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_OUT)
    print(f"💾 Live model saved → {MODEL_OUT}")

if __name__ == "__main__":
    main()
