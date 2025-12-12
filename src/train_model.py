import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "data/processed/events.csv"

def main():
    df = pd.read_csv(DATA_FILE)

    if df.empty or df["label_up"].nunique() < 2:
        print("❌ Not enough data to train model")
        return

    features = ["sentiment_score", "avg_confidence", "news_count"]
    X = df[features]
    y = df["label_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n✅ Accuracy:", accuracy_score(y_test, preds))
    print("\n📊 Classification report:\n")
    print(classification_report(y_test, preds))

    print("\n🧠 Feature importance (coefficients):")
    for f, c in zip(features, model.coef_[0]):
        print(f"{f:20s} {c:.4f}")

if __name__ == "__main__":
    main()
