import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/events.csv")

if df.empty:
    print("[WARN] No events available — skipping hit-rate plot")
    exit(0)

counts = df["label_up"].value_counts().sort_index()

plt.figure()
counts.plot(kind="bar")
plt.xticks([0,1], ["DOWN", "UP"], rotation=0)
plt.ylabel("Days")
plt.title("Next-day Direction Distribution")

plt.savefig("figures/hit_rate.png")
print("[OK] Saved figures/hit_rate.png")
