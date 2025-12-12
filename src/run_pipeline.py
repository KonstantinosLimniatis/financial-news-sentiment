import subprocess
import argparse

def run(cmd):
    print(f"\n▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    ticker = args.ticker.upper()

    run(f"python src/fetch_news_gdelt.py --ticker {ticker}")
    run("python src/sentiment_finbert.py")
    run(f"python src/fetch_prices.py --ticker {ticker}")
    run("python src/build_events.py")

if __name__ == "__main__":
    main()
