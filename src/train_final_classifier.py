import pandas as pd
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================
# Load weakly labeled training data
# ============================================

train_df = pd.read_csv("data/amazon_weak_labeled.csv")

train_df = train_df.dropna(
    subset=["customer_message", "weak_intent"]
).copy()

X_train = train_df["customer_message"]
y_train = train_df["weak_intent"]


# ============================================
# Load GOLDEN evaluation set
# ============================================

golden_df = pd.read_csv("data/golden_set.csv")

golden_df = golden_df.dropna(
    subset=["customer_message", "intent"]
).copy()

X_test = golden_df["customer_message"]
y_test = golden_df["intent"]


# ============================================
# Feature extraction
# ============================================

features = FeatureUnion([
    (
        "word_features",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=40000,
            sublinear_tf=True
        )
    ),
    (
        "character_features",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )
    )
])


# ============================================
# Classifier
# ============================================

model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ============================================
# Train
# ============================================

print("=" * 70)
print("TRAINING FINAL INTENT CLASSIFIER")
print("=" * 70)

print("Training examples:", len(X_train))
print("Golden test examples:", len(X_test))

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================
# Evaluate on GOLDEN SET
# ============================================

print("\nEvaluating on golden set...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================
# Save predictions
# ============================================

results = golden_df[
    [
        "customer_tweet_id",
        "customer_message",
        "intent"
    ]
].copy()

results["predicted_intent"] = predictions

results.to_csv(
    "results/intent_predictions.csv",
    index=False
)


# ============================================
# Save model
# ============================================

joblib.dump(
    model,
    "results/intent_classifier.joblib"
)

print("\nPredictions saved to:")
print("results/intent_predictions.csv")

print("\nModel saved to:")
print("results/intent_classifier.joblib")