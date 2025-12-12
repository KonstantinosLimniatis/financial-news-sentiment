import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_FILE = "data/processed/events_features.csv"
MODEL_PATH = "models/logistic_model.pkl"

FEATURES = [
    "sentiment_score",
    "avg_confidence",
    "news_count",
    "sentiment_3d_avg",
    "sentiment_5d_avg",
    "sentiment_volatility",
    "news_trend",
]

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("❌ Empty dataset")
        return

    if "label_up" not in df.columns:
        print("❌ label_up column missing")
        return

    if df["label_up"].nunique() < 2:
        print("❌ Not enough class diversity for training")
        return

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    X = df[FEATURES]
    y = df["label_up"]

    # --------------------------------------------------
    # Time-series cross validation
    # --------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=3)

    print("🔁 Time-series cross validation\n")

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Fold {i+1} accuracy: {acc:.2f}")

    # --------------------------------------------------
    # Train final model on full dataset
    # --------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # --------------------------------------------------
    # Model interpretability
    # --------------------------------------------------
    print("\n🧠 Final model coefficients:")
    for feature, coef in sorted(
        zip(FEATURES, model.coef_[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    ):
        print(f"{feature:25s} {coef:+.4f}")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n💾 Model saved → {MODEL_PATH}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
