import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

import yfinance as yf
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Live News Sentiment Trading Signal", layout="wide")
MODEL_PATH = "models/logistic_live.pkl"  # live model (preferred)
ALT_MODEL_PATH = "models/logistic_model.pkl"  # fallback


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model


def finbert_score_titles(titles: list[str]) -> pd.DataFrame:
    """
    Returns dataframe with columns: title, sentiment (-1/0/1), confidence (0-1)
    """
    if not titles:
        return pd.DataFrame(columns=["title", "sentiment", "confidence"])

    tokenizer, model = load_finbert()

    rows = []
    # small batching for speed
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = titles[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()

            # FinBERT label order is usually: [negative, neutral, positive]
            # We'll map to -1/0/+1
            for title, p in zip(batch, probs):
                idx = int(np.argmax(p))
                conf = float(np.max(p))
                if idx == 0:
                    sent = -1
                elif idx == 1:
                    sent = 0
                else:
                    sent = 1
                rows.append({"title": title, "sentiment": sent, "confidence": conf})

    return pd.DataFrame(rows)


def fetch_gdelt_news_simple(query: str, days: int, max_records: int = 250) -> pd.DataFrame:
    """
    Lightweight GDELT fetch using their DOC API via requests inside yfinance fallback logic
    Uses your existing src/fetch_news_gdelt.py logic if available; otherwise simple query.
    """
    try:
        # Prefer your project’s implementation (better + already tested by you)
        from src.fetch_news_gdelt import fetch_gdelt_articles, company_query_from_ticker, MAX_ARTICLES
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        company_q = company_query_from_ticker(query)
        arts = fetch_gdelt_articles(company_q, start, end, min(max_records, MAX_ARTICLES))
        used = company_q
        if not arts:
            arts = fetch_gdelt_articles(query, start, end, min(max_records, MAX_ARTICLES))
            used = query

        rows = []
        for a in arts:
            # seendate like 20251205121500
            sd = a.get("seendate")
            dt = end
            try:
                if isinstance(sd, str) and sd.isdigit() and len(sd) >= 14:
                    dt = datetime.strptime(sd[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                dt = end

            title = (a.get("title") or "").strip()
            url = (a.get("url") or "").strip()
            if title:
                rows.append({"date": dt, "title": title, "url": url})

        df = pd.DataFrame(rows).drop_duplicates(subset=["title", "url"]).head(max_records)
        df.attrs["used_query"] = used
        return df

    except Exception:
        # If import fails, return empty; UI will explain
        df = pd.DataFrame(columns=["date", "title", "url"])
        df.attrs["used_query"] = query
        return df


def aggregate_daily_signal(scored: pd.DataFrame) -> dict:
    """
    Build compact features from scored headlines.
    sentiment_score in [-1..+1] as (pos - neg) / n
    """
    if scored.empty:
        return {
            "news_count": 0,
            "sentiment_score": 0.0,
            "avg_confidence": 0.0,
            "pos": 0,
            "neg": 0,
            "neu": 0,
        }

    pos = int((scored["sentiment"] == 1).sum())
    neg = int((scored["sentiment"] == -1).sum())
    neu = int((scored["sentiment"] == 0).sum())
    n = len(scored)

    sentiment_score = (pos - neg) / max(n, 1)
    avg_conf = float(scored["confidence"].mean()) if "confidence" in scored else 0.0

    return {
        "news_count": int(n),
        "sentiment_score": float(sentiment_score),
        "avg_confidence": float(avg_conf),
        "pos": pos,
        "neg": neg,
        "neu": neu,
    }


def load_any_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), MODEL_PATH
    if os.path.exists(ALT_MODEL_PATH):
        return joblib.load(ALT_MODEL_PATH), ALT_MODEL_PATH
    return None, None


def predict_prob(model, features: dict) -> float:
    """
    Build X with the model’s expected feature names.
    If some are missing, we fill with safe defaults.
    """
    # defaults for engineered fields
    defaults = {
        "sentiment_score": features.get("sentiment_score", 0.0),
        "avg_confidence": features.get("avg_confidence", 0.0),
        "news_count": features.get("news_count", 0),
        "sentiment_3d_avg": features.get("sentiment_score", 0.0),
        "sentiment_5d_avg": features.get("sentiment_score", 0.0),
        "sentiment_volatility": 0.0,
        "news_trend": 0.0,
    }

    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    else:
        # fallback
        cols = [
            "sentiment_score",
            "avg_confidence",
            "news_count",
            "sentiment_3d_avg",
            "sentiment_5d_avg",
            "sentiment_volatility",
            "news_trend",
        ]

    X = pd.DataFrame([{c: defaults.get(c, 0.0) for c in cols}])
    proba = float(model.predict_proba(X)[0][1])
    return proba


def make_bias_label(prob_up: float) -> tuple[str, str]:
    """
    UI-friendly label regardless of trade decision.
    """
    if prob_up >= 0.60:
        return "Bullish bias", "🟢"
    if prob_up <= 0.40:
        return "Bearish bias", "🔴"
    return "Neutral / uncertain", "🟡"


def make_trade_label(prob_up: float) -> tuple[str, str]:
    """
    Keep trade thresholds conservative but not “dead”.
    """
    if prob_up >= 0.65:
        return "BUY", "HIGH"
    if prob_up >= 0.55:
        return "BUY", "MEDIUM"
    return "NO TRADE", "LOW"


def fetch_last_close_price(ticker: str, lookback_days: int = 10) -> tuple[float | None, pd.Timestamp | None]:
    """
    Gets last available close (works even if market is closed).
    """
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=lookback_days)
    df = yf.download(ticker, start=start.date(), end=(end + pd.Timedelta(days=1)).date(), progress=False)
    if df is None or df.empty:
        return None, None
    df = df.reset_index()
    # handle multiindex columns
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    last = df.iloc[-1]
    return float(last["Close"]), pd.to_datetime(last["Date"])


# -----------------------------
# UI
# -----------------------------
st.title("📈 Live News Sentiment Trading Signal")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Stock ticker", value="GOOGL").strip().upper()
    days = st.slider("News lookback (days)", min_value=1, max_value=30, value=7)
    max_news = st.slider("Max headlines", min_value=20, max_value=250, value=120, step=10)
    run_btn = st.button("Run Live Signal", type="primary")

st.caption("Demo: News → FinBERT sentiment → (optional) ML probability. Not financial advice.")

if run_btn:
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    # 1) Fetch news
    with st.spinner("Fetching news (GDELT)…"):
        news_df = fetch_gdelt_news_simple(ticker, days=days, max_records=max_news)
        used_query = news_df.attrs.get("used_query", ticker)

    # 2) Score sentiment
    with st.spinner("Scoring sentiment (FinBERT)…"):
        scored = finbert_score_titles(news_df["title"].tolist() if not news_df.empty else [])

    # 3) Aggregate
    agg = aggregate_daily_signal(scored)

    # 4) Price (last close)
    with st.spinner("Fetching last close price…"):
        last_close, last_close_dt = fetch_last_close_price(ticker)

    # 5) Model probability (if model exists)
    model, model_path = load_any_model()
    if model is not None:
        prob_up = predict_prob(model, agg)
    else:
        # fallback heuristic probability from sentiment_score (so UI always works)
        # map sentiment_score [-1..+1] into prob [0.2..0.8]
        prob_up = float(np.clip(0.5 + 0.3 * agg["sentiment_score"], 0.2, 0.8))

    bias, bias_icon = make_bias_label(prob_up)
    trade, conf = make_trade_label(prob_up)

    # -----------------------------
    # Layout
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Ticker", ticker)
    c2.metric("Headlines analyzed", agg["news_count"])
    c3.metric("Sentiment score", f"{agg['sentiment_score']:.2f}")
    if last_close is not None and last_close_dt is not None:
        c4.metric("Last close", f"{last_close:.2f}", help=f"Last available close: {last_close_dt.date()}")
    else:
        c4.metric("Last close", "N/A")

    st.divider()

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Result")

        # Probability bar
        st.write(f"**Model probability (up next day):** `{prob_up:.2f}`")
        st.progress(min(max(prob_up, 0.0), 1.0))

        # Friendly summary
        st.write(f"**Bias:** {bias_icon} {bias}")
        st.write(f"**Trade decision:** `{trade}` (confidence: `{conf}`)")

        if model_path:
            st.caption(f"Model used: `{model_path}`")
        else:
            st.caption("No trained model found → using a safe heuristic so the demo still works.")

        # Explain “why no trade”
        if trade == "NO TRADE":
            st.info(
                "No-trade is NORMAL. It means the system doesn't see enough edge.\n\n"
                f"- Probability is only **{prob_up:.2f}**\n"
                f"- News may be mixed/uncertain\n"
                "Tip: try increasing lookback days (e.g. 14–30) or a different ticker."
            )

        st.subheader("Sentiment breakdown")
        b1, b2, b3 = st.columns(3)
        b1.metric("Positive", agg["pos"])
        b2.metric("Neutral", agg["neu"])
        b3.metric("Negative", agg["neg"])

        # Pie-like bar chart (simple, stable)
        counts = pd.Series({"Positive": agg["pos"], "Neutral": agg["neu"], "Negative": agg["neg"]})
        fig = plt.figure()
        counts.plot(kind="bar")
        plt.title("Headline sentiment counts")
        plt.xlabel("")
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Top headlines (sample)")

        if news_df.empty or scored.empty:
            st.warning("No headlines returned or scored. Try a different ticker or increase days.")
        else:
            view = news_df.copy()
            view["sentiment"] = scored["sentiment"].values
            view["confidence"] = scored["confidence"].values
            view = view.sort_values("confidence", ascending=False).head(12)

            def sent_label(x: int) -> str:
                return "POS" if x == 1 else ("NEG" if x == -1 else "NEU")

            for _, row in view.iterrows():
                lab = sent_label(int(row["sentiment"]))
                st.write(f"**[{lab}]** ({row['confidence']:.2f}) {row['title']}")
                if isinstance(row.get("url"), str) and row["url"]:
                    st.caption(row["url"])

    st.divider()

    # Save artifacts for your backend structure (optional but helpful)
    os.makedirs(f"data/raw/{ticker}", exist_ok=True)
    os.makedirs(f"data/processed/{ticker}", exist_ok=True)

    if not news_df.empty:
        news_df.to_csv(f"data/raw/{ticker}/news.csv", index=False)
    if not scored.empty:
        scored.to_csv(f"data/processed/{ticker}/news_scored.csv", index=False)

    st.success("Done. The UI is showing meaningful output for the user (even when it’s NO TRADE).")

else:
    st.info("Pick a ticker and click **Run Live Signal**.")
