import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    in_file = f"data/processed/{ticker}/events.csv"
    out_file = f"figures/{ticker}_scatter.png"

    if not os.path.exists(in_file):
        print(f"[WARN] No events file for {ticker}")
        return

    df = pd.read_csv(in_file)

    if df.empty:
        print(f"[WARN] Empty events for {ticker}")
        return

    plt.figure()
    plt.scatter(df["sentiment_score"], df["next_day_return"])
    plt.axhline(0)
    plt.xlabel("Sentiment score")
    plt.ylabel("Next-day return (%)")
    plt.title(f"{ticker} — Sentiment vs Return")

    plt.savefig(out_file)
    plt.close()

    print(f"[OK] Saved {out_file}")

if __name__ == "__main__":
    main()

