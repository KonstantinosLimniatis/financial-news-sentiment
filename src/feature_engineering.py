import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g. GOOGL)")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    in_file = f"data/processed/{ticker}/events.csv"
    out_file = f"data/processed/{ticker}/events_features.csv"

    if not os.path.exists(in_file):
        print(f"❌ Events file not found for {ticker}")
        return

    df = pd.read_csv(in_file)

    if df.empty:
        print(f"❌ Empty events file for {ticker}")
        return

    # --------------------------------------------------
    # Sort by date
    # --------------------------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    df["sentiment_3d_avg"] = df["sentiment_score"].rolling(3).mean()
    df["sentiment_5d_avg"] = df["sentiment_score"].rolling(5).mean()

    df["sentiment_volatility"] = df["sentiment_score"].rolling(5).std()

    df["news_trend"] = df["news_count"].diff()

    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs(f"data/processed/{ticker}", exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"✅ Saved engineered features → {out_file}")
    print(df.head())

if __name__ == "__main__":
    main()
