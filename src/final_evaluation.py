import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 1. LOAD DATA
# ============================================================

training = pd.read_csv("data/amazon_weak_labeled.csv", encoding="utf-8")
golden = pd.read_csv(
    "data/golden_set.csv",
    encoding="latin1"
)

print("Training rows:", len(training))
print("Golden test rows:", len(golden))


# ============================================================
# 2. REMOVE GOLDEN SET FROM TRAINING
# ============================================================

golden_ids = set(golden["customer_tweet_id"].astype(str))

training["customer_tweet_id"] = training["customer_tweet_id"].astype(str)

training = training[
    ~training["customer_tweet_id"].isin(golden_ids)
].copy()

print("Training rows after removing golden set:", len(training))


# ============================================================
# 3. PREPARE TEXT
# ============================================================

X_train = training["customer_message"].fillna("").astype(str)
y_train = training["weak_intent"].astype(str)

X_test = golden["customer_message"].fillna("").astype(str)
y_test = golden["intent"].astype(str)


# ============================================================
# 4. WORD + CHARACTER TF-IDF
# ============================================================

word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True,
    min_df=2
)

char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=20000,
    sublinear_tf=True,
    min_df=2
)

features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf)
])


# ============================================================
# 5. TRANSFORM TRAINING DATA
# ============================================================

print("\nBuilding TF-IDF features...")

X_train_features = features.fit_transform(X_train)
X_test_features = features.transform(X_test)

print("Training feature shape:", X_train_features.shape)
print("Test feature shape:", X_test_features.shape)


# ============================================================
# 6. TRAIN CLASSIFIER
# ============================================================

print("\nTraining classifier...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_features, y_train)


# ============================================================
# 7. PREDICT GOLDEN SET
# ============================================================

predictions = model.predict(X_test_features)


# ============================================================
# 8. EVALUATE
# ============================================================

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 70)
print("FINAL INTENT CLASSIFICATION RESULTS")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

labels = sorted(set(y_test) | set(predictions))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(cm_df)


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

results = golden.copy()

results["predicted_intent"] = predictions

results.to_csv(
    "results/final_intent_predictions.csv",
    index=False
)

print("\nSaved:")
print("results/final_intent_predictions.csv")


# ============================================================
# 11. SHOW WRONG PREDICTIONS
# ============================================================

results["correct"] = (
    results["intent"] == results["predicted_intent"]
)

wrong = results[results["correct"] == False]

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print("\nCorrect predictions:", len(results) - len(wrong))
print("Wrong predictions:", len(wrong))

print("\nFirst 20 wrong predictions:")

print(
    wrong[
        [
            "customer_message",
            "intent",
            "predicted_intent"
        ]
    ].head(20).to_string(index=False)
)