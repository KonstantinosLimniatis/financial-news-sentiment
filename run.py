import subprocess
import pandas as pd
import os

def run(cmd):
    print(f"\n▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("\n=== Financial News Sentiment Analyzer ===\n")

    ticker = input("Enter stock ticker (e.g. GOOGL, META, NVDA): ").upper()
    print(f"\n✅ Running analysis for {ticker}")

    run(f"python src/fetch_news_gdelt.py --ticker {ticker}")
    run("python src/sentiment_finbert.py")
    run(f"python src/fetch_prices.py --ticker {ticker}")
    run("python src/build_events.py")

    events_file = "data/processed/events.csv"
    if not os.path.exists(events_file):
        print("\n❌ No events file found")
        return

    df = pd.read_csv(events_file)
    if df.empty:
        print("\n⚠️ No events generated for this stock.")
        print("Possible reasons:")
        print("- Low news coverage")
        print("- No overlap between news and trading days")
        print("- Short time window")
        return

    print("\n📊 Generating plots...")
    run("python src/plot_scatter.py")
    run("python src/plot_hit_rate.py")
    run("python src/plot_equity.py")

    print("\n🎉 Done! Results in ./figures")

if __name__ == "__main__":
    main()
