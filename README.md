# 📈 Financial News Sentiment Trading System

A research & demo project that analyzes **financial news sentiment** for stocks and produces **probabilistic trading signals** using NLP and machine learning.

⚠️ **Not financial advice. Educational / research purposes only.**

---

## 🚀 What This Project Does (High Level)

This system follows the pipeline:

**News → NLP Sentiment → Feature Engineering → ML Probability → Trade Signal**

For a given stock ticker (e.g. `GOOGL`, `META`, `NVDA`):

1. Fetches recent financial news articles (GDELT API)
2. Scores headlines using **FinBERT** (financial sentiment model)
3. Aggregates sentiment per day
4. Engineers time-series features
5. Uses a **logistic regression model** to estimate the probability that the stock goes **up the next trading day**
6. Outputs a **LIVE signal**:
   - BUY (High / Medium confidence)
   - NO TRADE (Low confidence)

The system is **designed to say NO TRADE most of the time** — this is intentional.

---

## 🧠 Why “NO TRADE” Is Normal

Markets are noisy.  
News sentiment alone rarely provides strong predictive edge.

This project uses **probability thresholds**:
- `>= 0.65` → BUY (High confidence)
- `>= 0.55` → BUY (Medium confidence)
- `< 0.55` → NO TRADE

👉 If the system says **NO TRADE**, it means:
- News sentiment is mixed or weak
- The model does not see sufficient statistical edge
- Doing nothing is the safest decision

This is a **feature, not a bug**.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/KonstantinosLimniatis/financial-news-sentiment.git
cd financial-news-sentiment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


▶️ How to Run (CLI)
Step 1: Fetch news
python src/fetch_news_gdelt.py --ticker GOOGL --days 7

Step 2: Score sentiment (FinBERT)
python src/sentiment_finbert.py --ticker GOOGL

Step 3: Fetch stock prices
python src/fetch_prices.py --ticker GOOGL

Step 4: Build daily events
python src/build_events.py --ticker GOOGL

Step 5: Engineer features
python src/feature_engineering.py --ticker GOOGL

Step 6: Train model (optional)
python src/train_model_v2.py

Step 7: Generate live signal
python -m src.live_signal --ticker GOOGL

🖥️ Frontend (Streamlit Demo)

Run the web interface:

streamlit run app.py --server.fileWatcherType none


Open browser at:

http://localhost:8501

📊 Plots

Generate analytics plots:

python src/plot_scatter.py
python src/plot_hit_rate.py
python src/plot_equity.py


Saved under:

figures/

🧑‍💻 Author

Konstantinos Limniatis
Computer Science & Telecommunications
Harokopio University of Athens
