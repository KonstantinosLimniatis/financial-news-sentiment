import yfinance as yf
import re

def ticker_to_company(ticker: str) -> str:
    ticker = ticker.upper()
    stock = yf.Ticker(ticker)
    info = stock.info

    name = info.get("shortName") or info.get("longName")
    if not name:
        raise ValueError(f"Could not resolve company name for {ticker}")

    # aggressive cleanup
    name = re.sub(r",.*", "", name)          # remove commas and after
    name = re.sub(r"\b(Inc|Corp|Ltd|PLC|S\.A\.|Holdings|Group)\b", "", name)
    name = name.replace("Platforms", "")
    name = name.strip()

    return name
