import pandas as pd

LIVE_FEATURES = [
    "sentiment_score",
    "avg_confidence",
    "news_count",
]

def build_live_features(news_scored_file: str) -> pd.DataFrame:
    df = pd.read_csv(news_scored_file)

    if df.empty:
        raise ValueError("Empty news file")

    # Aggregate TODAY
    sentiment_score = df["sentiment"].mean()
    avg_confidence = df["confidence"].mean()
    news_count = len(df)

    X = pd.DataFrame([{
        "sentiment_score": sentiment_score,
        "avg_confidence": avg_confidence,
        "news_count": news_count,
    }])

    return X[LIVE_FEATURES]
