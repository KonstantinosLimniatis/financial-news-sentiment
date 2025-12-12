import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

LABEL_MAP = {
    0: -1,  # negative
    1: 0,   # neutral
    2: 1,   # positive
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g. GOOGL)")
    parser.add_argument("--max_rows", type=int, default=80, help="Max headlines to score")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    max_rows = args.max_rows

    in_file = f"data/raw/{ticker}/news.csv"
    out_dir = f"data/processed/{ticker}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/news_scored.csv"

    if not os.path.exists(in_file):
        print(f"[ERROR] Missing input file: {in_file}")
        return

    df = pd.read_csv(in_file)
    if df.empty:
        print("[WARN] No news to score")
        df.to_csv(out_file, index=False)
        return

    df = df.head(max_rows)

    print("[INFO] Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    sentiments = []
    confidences = []

    for title in tqdm(df["title"], desc="Scoring headlines"):
        inputs = tokenizer(
            title,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()

        label_idx = int(torch.argmax(probs))
        sentiment = LABEL_MAP[label_idx]
        confidence = float(probs[label_idx])

        sentiments.append(sentiment)
        confidences.append(confidence)

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    df.to_csv(out_file, index=False)
    print(f"[OK] Saved sentiment scores → {out_file}")
    print(df.head())

if __name__ == "__main__":
    main()
