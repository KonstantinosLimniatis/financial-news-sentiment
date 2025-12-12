import joblib
import pandas as pd

FEATURES = [
    "sentiment_score",
    "avg_confidence",
    "news_count",
    "sentiment_3d_avg",
    "sentiment_5d_avg",
    "sentiment_volatility",
    "news_trend",
]

def explain_signal(model_path, X_live):
    model = joblib.load(model_path)

    coefs = model.coef_[0]
    values = X_live.iloc[0].values

    contributions = {
        f: coefs[i] * values[i]
        for i, f in enumerate(FEATURES)
    }

    sorted_contrib = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    positives = [(f, v) for f, v in sorted_contrib if v > 0][:3]
    negatives = [(f, v) for f, v in sorted_contrib if v < 0][:3]

    return positives, negatives
