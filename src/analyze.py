import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAP = {"negative": -1, "neutral": 0, "positive": 1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/raw/prices.csv")
    ap.add_argument("--news_scored", default="data/processed/news_scored.csv")
    ap.add_argument("--out", default="data/processed/merged.csv")
    ap.add_argument("--fig", default="figures/scatter.png")
    args = ap.parse_args()

    # READ prices FIRST
    prices = pd.read_csv(args.prices)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values("date").copy()

    # FIX numeric columns (MUST be AFTER read_csv)
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    prices["next_close"] = prices["Close"].shift(-1)
    prices["next_day_return"] = (prices["next_close"] / prices["Close"] - 1.0) * 100.0

    # READ news
    news = pd.read_csv(args.news_scored)
    news["date"] = pd.to_datetime(news["date"]).dt.date

    MAP = {"negative": -1, "neutral": 0, "positive": 1}
    news["score_basic"] = news["sentiment_basic"].map(MAP)
    news["score_adv"] = news["sentiment_adv"].map(MAP)

    daily = news.groupby("date", as_index=False).agg(
        score_basic=("score_basic", "mean"),
        score_adv=("score_adv", "mean"),
        confidence_adv=("confidence_adv", "mean"),
        n_news=("title", "count"),
    )

    # Align news dates to next available trading day
    prices_sorted = prices.sort_values("date")[["date", "next_day_return"]]

    daily["aligned_date"] = daily["date"].apply(
        lambda d: prices_sorted[prices_sorted["date"] >= d]["date"].min()
    )

    merged = pd.merge(
        daily,
        prices_sorted,
        left_on="aligned_date",
        right_on="date",
        how="inner",
    )


    merged = merged.dropna(subset=["next_day_return"])
    merged.to_csv(args.out, index=False)

    corr_basic = merged["score_basic"].corr(merged["next_day_return"])
    corr_adv = merged["score_adv"].corr(merged["next_day_return"])

    print(f"[RESULT] Pearson corr (Basic):    {corr_basic:.4f}")
    print(f"[RESULT] Pearson corr (Advanced): {corr_adv:.4f}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(merged["score_adv"], merged["next_day_return"])
    plt.xlabel("Daily sentiment score (Advanced)")
    plt.ylabel("Next-day return (%)")
    plt.title("Sentiment vs Next-day Return")
    plt.grid(True, alpha=0.3)
    plt.savefig(args.fig, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved figure: {args.fig}")

if __name__ == "__main__":
    main()
