import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# LOAD GOLDEN DATA
# ============================================================

df = pd.read_csv("data/golden_set.csv")

df = df.dropna(
    subset=["customer_message", "intent"]
).copy()

X = df["customer_message"]
y = df["intent"]


# ============================================================
# WORD + CHARACTER FEATURES
# ============================================================

features = FeatureUnion([
    (
        "word_features",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
            sublinear_tf=True
        )
    ),

    (
        "character_features",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=20000,
            sublinear_tf=True
        )
    )
])


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print("Running 5-fold cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("\n")
print("=" * 60)
print("IMPROVED INTENT CLASSIFIER")
print("=" * 60)

print("Fold accuracies:")

for i, score in enumerate(scores, start=1):
    print(
        f"Fold {i}: {score * 100:.2f}%"
    )

print(
    "\nMean accuracy:",
    round(scores.mean() * 100, 2),
    "%"
)

print(
    "Standard deviation:",
    round(scores.std() * 100, 2),
    "%"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining final model...")

model.fit(X, y)

print("Final model trained.")


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    "results/intent_classifier.joblib"
)

print("\nModel saved to:")
print("results/intent_classifier.joblib")