import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading datasets...")

golden = pd.read_csv("data/golden_set.csv")
historical = pd.read_csv("data/amazon_training_data.csv")

golden = golden.dropna(
    subset=["customer_message", "intent"]
).copy()

historical = historical.dropna(
    subset=["customer_message", "amazon_reply"]
).copy()

print("Golden examples:", len(golden))
print("Historical conversations:", len(historical))


# ============================================================
# 2. TRAIN INTENT CLASSIFIER
# ============================================================

print("\nTraining intent classifier...")

intent_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

intent_model.fit(
    golden["customer_message"],
    golden["intent"]
)

print("Classifier trained.")


# ============================================================
# 3. BUILD RETRIEVAL INDEX
# ============================================================

print("\nBuilding retrieval index...")

retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000
)

retrieval_matrix = retrieval_vectorizer.fit_transform(
    historical["customer_message"]
)

print("Retrieval index ready.")


# ============================================================
# 4. EVALUATE INTENT CLASSIFICATION
# ============================================================

predicted_intents = intent_model.predict(
    golden["customer_message"]
)

true_intents = golden["intent"]

accuracy = accuracy_score(
    true_intents,
    predicted_intents
)

print("\n")
print("=" * 60)
print("INTENT CLASSIFICATION")
print("=" * 60)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        true_intents,
        predicted_intents,
        zero_division=0
    )
)


# ============================================================
# 5. RETRIEVAL EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)

similarities = []

for message in golden["customer_message"]:

    query_vector = retrieval_vectorizer.transform(
        [message]
    )

    scores = cosine_similarity(
        query_vector,
        retrieval_matrix
    ).flatten()

    best_score = float(
        np.max(scores)
    )

    similarities.append(best_score)

print(
    "Average similarity:",
    round(np.mean(similarities), 4)
)

print(
    "Median similarity:",
    round(np.median(similarities), 4)
)

print(
    "Minimum similarity:",
    round(np.min(similarities), 4)
)

print(
    "Maximum similarity:",
    round(np.max(similarities), 4)
)


# ============================================================
# 6. ESCALATION EVALUATION
# ============================================================

def decide_escalation(
    intent,
    similarity,
    customer_message
):

    message = customer_message.lower()

    if intent == "account_security":
        return True

    if intent == "seller_issue":
        return True

    if similarity < 0.20:
        return True

    keywords = [
        "fraud",
        "scam",
        "hacked",
        "stolen",
        "unauthorized",
        "chargeback",
        "lawsuit",
        "legal",
        "police"
    ]

    for keyword in keywords:

        if keyword in message:
            return True

    return False


predicted_escalation = []

for i, row in golden.iterrows():

    intent = predicted_intents[
        list(golden.index).index(i)
    ]

    similarity = similarities[
        list(golden.index).index(i)
    ]

    result = decide_escalation(
        intent,
        similarity,
        row["customer_message"]
    )

    predicted_escalation.append(
        "yes" if result else "no"
    )


true_escalation = golden[
    "should_escalate"
].astype(str).str.lower().tolist()


escalation_accuracy = accuracy_score(
    true_escalation,
    predicted_escalation
)

print("\n")
print("=" * 60)
print("ESCALATION EVALUATION")
print("=" * 60)

print(
    "Escalation accuracy:",
    round(escalation_accuracy * 100, 2),
    "%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        true_escalation,
        predicted_escalation,
        labels=["no", "yes"]
    )
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results = golden.copy()

results["predicted_intent"] = predicted_intents
results["retrieval_similarity"] = similarities
results["predicted_escalation"] = predicted_escalation

results.to_csv(
    "results/evaluation_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/evaluation_results.csv")