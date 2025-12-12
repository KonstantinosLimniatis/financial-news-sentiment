import argparse
import pandas as pd
import yfinance as yf
from datetime import timedelta
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    ticker = args.ticker.upper()

    RAW_DIR = Path(f"data/raw/{ticker}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    OUT_FILE = RAW_DIR / "prices.csv"
    NEWS_FILE = Path(f"data/processed/{ticker}/news_scored.csv")

    # --------------------------------------------------
    # Load news dates
    # --------------------------------------------------
    if NEWS_FILE.exists():
        news = pd.read_csv(NEWS_FILE)
        news["date"] = pd.to_datetime(news["date"], errors="coerce")
        news_dates = news["date"].dropna()
    else:
        news_dates = pd.Series(dtype="datetime64[ns]")

    # --------------------------------------------------
    # Determine date range
    # --------------------------------------------------
    if news_dates.empty:
        print("[WARN] No news dates — fetching last 30 days prices")
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=30)
    else:
        start = news_dates.min().date()
        end = (news_dates.max() + timedelta(days=1)).date()

    print(f"[INFO] Fetching prices {start} → {end}")

    # --------------------------------------------------
    # Download prices
    # --------------------------------------------------
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
    )

    if df.empty:
        print("[WARN] No price data returned")
        pd.DataFrame().to_csv(OUT_FILE, index=False)
        return

    df = df.reset_index()

    # --------------------------------------------------
    # Clean columns
    # --------------------------------------------------
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "Open", "High", "Low", "Close", "Volume"]

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    df.to_csv(OUT_FILE, index=False)
    print(f"[OK] Saved prices → {OUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()
