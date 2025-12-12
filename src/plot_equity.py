import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/events.csv")

if df.empty:
    print("[WARN] No events available — skipping equity curve")
    exit(0)

df["strategy_return"] = df["next_day_return"] * (df["sentiment_score"] > 0)
df["equity"] = (1 + df["strategy_return"] / 100).cumprod()

plt.figure()
plt.plot(df["date"], df["equity"])
plt.xlabel("Date")
plt.ylabel("Equity (start=1)")
plt.title("Simple Sentiment Strategy Equity Curve")

plt.savefig("figures/equity_curve.png")
print("[OK] Saved figures/equity_curve.png")
