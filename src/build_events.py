import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    ticker = args.ticker.upper()

    news_file = f"data/processed/{ticker}/news_scored.csv"
    prices_file = f"data/raw/{ticker}/prices.csv"
    out_dir = f"data/processed/{ticker}"
    out_file = f"{out_dir}/events.csv"

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(news_file) or not os.path.exists(prices_file):
        print("[INFO] Missing inputs — skipping events (live mode)")
        pd.DataFrame().to_csv(out_file, index=False)
        return

    news = pd.read_csv(news_file)
    prices = pd.read_csv(prices_file)

    if news.empty or prices.empty:
        print("[INFO] No data for backtest yet (live mode)")
        pd.DataFrame().to_csv(out_file, index=False)
        return

    news["date"] = pd.to_datetime(news["date"]).dt.date
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    daily = (
        news.groupby("date")
        .agg(
            sentiment_score=("sentiment", "mean"),
            avg_confidence=("confidence", "mean"),
            news_count=("sentiment", "count"),
        )
        .reset_index()
    )

    prices["next_close"] = prices["Close"].shift(-1)
    prices["next_day_return"] = (
        (prices["next_close"] / prices["Close"] - 1) * 100
    )

    df = pd.merge(daily, prices[["date", "next_day_return"]], on="date", how="inner")
    df = df.dropna()

    df["label_up"] = (df["next_day_return"] > 0).astype(int)

    df.to_csv(out_file, index=False)
    print(f"[OK] Saved events → {out_file}")

if __name__ == "__main__":
    main()

