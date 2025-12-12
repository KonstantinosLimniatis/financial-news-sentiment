import argparse
import json
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

load_dotenv()

PROMPT_BASIC = """Classify the sentiment of the following financial news headline about {ticker}.
Return ONLY valid JSON with keys:
- sentiment: one of ["positive","negative","neutral"]
- reasoning: short (max 25 words)

Headline: {headline}
"""

PROMPT_ADVANCED = """You are a Senior Financial Analyst.
Task: analyze the headline ONLY for financially-material information for {ticker}.
Ignore hype, vague claims, memes, unrelated macro noise unless directly tied to the company.
Return ONLY valid JSON with keys:
- sentiment: one of ["positive","negative","neutral"]
- confidence_score: float in [0,1]
- drivers: array of 1-3 short bullet-like strings (financial drivers)
- reasoning: short (max 35 words)

Headline: {headline}
"""

def call_model(client: OpenAI, model: str, prompt: str) -> dict:
    # Responses API (recommended) :contentReference[oaicite:4]{index=4}
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    text = resp.output_text.strip()
    return json.loads(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--news_csv", default="data/raw/news.csv")
    ap.add_argument("--out", default="data/processed/news_scored.csv")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=80, help="how many headlines to score (cost control)")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in environment (.env).")

    df = pd.read_csv(args.news_csv)
    df = df.dropna(subset=["title"]).copy()
    df = df.sort_values("published_dt", ascending=True)

    # Cost control: take most recent N (or all if smaller)
    if args.limit and len(df) > args.limit:
        df = df.tail(args.limit).copy()

    client = OpenAI()

    basic_sent, basic_reason = [], []
    adv_sent, adv_conf, adv_drivers, adv_reason = [], [], [], []

    for title in tqdm(df["title"].tolist(), desc="Scoring headlines"):
        # Basic
        out_a = call_model(client, args.model, PROMPT_BASIC.format(ticker=args.ticker, headline=title))
        basic_sent.append(out_a.get("sentiment"))
        basic_reason.append(out_a.get("reasoning"))

        # Advanced
        out_b = call_model(client, args.model, PROMPT_ADVANCED.format(ticker=args.ticker, headline=title))
        adv_sent.append(out_b.get("sentiment"))
        adv_conf.append(out_b.get("confidence_score"))
        adv_drivers.append(json.dumps(out_b.get("drivers", []), ensure_ascii=False))
        adv_reason.append(out_b.get("reasoning"))

    df["sentiment_basic"] = basic_sent
    df["reason_basic"] = basic_reason
    df["sentiment_adv"] = adv_sent
    df["confidence_adv"] = adv_conf
    df["drivers_adv"] = adv_drivers
    df["reason_adv"] = adv_reason

    df.to_csv(args.out, index=False)
    print(f"[OK] Saved scored news: {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
