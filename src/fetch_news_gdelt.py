import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

# GDELT 2.1 DOCS:
# https://blog.gdeltproject.org/gdelt-2-1-api-debuts/
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

MAX_ARTICLES = 250


def company_query_from_ticker(ticker: str) -> str:
    """
    Try to get a reasonable company name for better news recall.
    Fallback to ticker if we can't.
    """
    try:
        info = yf.Ticker(ticker).get_info()
        name = info.get("longName") or info.get("shortName")
        if name and isinstance(name, str) and len(name) > 2:
            return name
    except Exception:
        pass
    return ticker


def fetch_gdelt_articles(
    query: str,
    start: datetime,
    end: datetime,
    max_records: int,
) -> list[dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "maxrecords": str(max_records),
        "sort": "HybridRel",
    }

    r = requests.get(GDELT_DOC_API, params=params, timeout=30)

    # Sometimes GDELT returns HTML (rate limit / temporary issues)
    if "application/json" not in r.headers.get("Content-Type", ""):
        return []

    data = r.json()
    return data.get("articles", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g. GOOGL, NVDA)")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback window in days (default: 30)",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    days = args.days

    out_dir = f"data/raw/{ticker}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/news.csv"

    # --------------------------------------------------
    # 🔑 OPTION A:
    # Fetch news only up to YESTERDAY (UTC)
    # --------------------------------------------------
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    company_query = company_query_from_ticker(ticker)

    print(f"[INFO] Fetching GDELT news for {company_query} ({ticker})")
    print(f"[INFO] Date range: {start_date.date()} → {end_date.date()}")

    # Try company name first; fallback to ticker
    articles = fetch_gdelt_articles(
        company_query,
        start_date,
        end_date,
        MAX_ARTICLES,
    )
    used_query = company_query

    if not articles:
        articles = fetch_gdelt_articles(
            ticker,
            start_date,
            end_date,
            MAX_ARTICLES,
        )
        used_query = ticker

    print(f"[INFO] Found {len(articles)} articles using query: '{used_query}'")

    rows = []
    for a in articles:
        # Prefer seendate (YYYYMMDDHHMMSS)
        dt = a.get("seendate")

        try:
            if isinstance(dt, str) and dt.isdigit() and len(dt) >= 14:
                date = datetime.strptime(
                    dt[:14],
                    "%Y%m%d%H%M%S",
                ).replace(tzinfo=timezone.utc)
            else:
                date = end_date
        except Exception:
            date = end_date

        title = a.get("title") or ""
        url = a.get("url") or ""

        if title.strip():
            rows.append(
                {
                    "date": date.isoformat(),
                    "title": title.strip(),
                    "url": url,
                }
            )

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["title", "url"])
        .head(MAX_ARTICLES)
    )

    df.to_csv(out_file, index=False)

    print(f"[OK] Saved {len(df)} news → {out_file}")
    if not df.empty:
        print(df.head())


if __name__ == "__main__":
    main()
