import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_FILE = "data/processed/events.csv"

THRESHOLD = 0.6

def main():
    df = pd.read_csv(DATA_FILE)

    if df.empty or df["label_up"].nunique() < 2:
        print("❌ Not enough data")
        return

    features = ["sentiment_score", "avg_confidence", "news_count"]
    X = df[features]
    y = df["label_up"]

    model = LogisticRegression()
    model.fit(X, y)

    df["prob_up"] = model.predict_proba(X)[:, 1]

    # Apply filter
    trades = df[df["prob_up"] >= THRESHOLD].copy()

    if trades.empty:
        print("⚠️ No trades passed the probability filter")
        return

    hit_rate = trades["label_up"].mean()
    avg_return = trades["next_day_return"].mean()

    print(f"\n📊 Trades taken: {len(trades)}")
    print(f"🎯 Hit rate:     {hit_rate:.2f}")
    print(f"📈 Avg return:   {avg_return:.2f}%")

    print("\nSample trades:")
    print(trades[[
        "date", "sentiment_score", "avg_confidence",
        "prob_up", "next_day_return", "label_up"
    ]].head())

if __name__ == "__main__":
    main()
