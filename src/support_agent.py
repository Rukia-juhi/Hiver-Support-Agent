import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading data...")

golden = pd.read_csv("data/golden_set.csv")
historical = pd.read_csv("data/amazon_training_data.csv")

print("Historical conversations:", len(historical))
print("Golden examples:", len(golden))


# ============================================================
# 2. TRAIN INTENT CLASSIFIER
# ============================================================

print("\nTraining intent classifier...")

X_train = golden["customer_message"]
y_train = golden["intent"]

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

intent_model.fit(X_train, y_train)

print("Intent classifier trained.")


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
    historical["customer_message"].fillna("")
)

print("Retrieval index ready.")


# ============================================================
# 4. ESCALATION LOGIC
# ============================================================

def decide_escalation(
    intent,
    similarity,
    customer_message
):

    message = customer_message.lower()

    # Security/account issues
    if intent == "account_security":
        return True, "Account or security issue requires human review."

    # Seller disputes
    if intent == "seller_issue":
        return True, "Seller-related issue requires human review."

    # Very low retrieval confidence
    if similarity < 0.20:
        return True, "No sufficiently similar historical case was found."

    # High-risk keywords
    escalation_keywords = [
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

    for keyword in escalation_keywords:
        if keyword in message:
            return True, (
                "Potentially high-risk issue "
                "requires human review."
            )

    return False, "Historical case provides sufficient guidance."


# ============================================================
# 5. RETRIEVE SIMILAR CASE
# ============================================================

def retrieve_case(customer_message):

    query_vector = retrieval_vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        retrieval_matrix
    ).flatten()

    best_index = np.argmax(similarities)

    similarity = float(
        similarities[best_index]
    )

    return (
        similarity,
        historical.iloc[best_index]
    )


# ============================================================
# 6. COMPLETE SUPPORT AGENT
# ============================================================

def support_agent(customer_message):

    # Predict intent
    intent = intent_model.predict(
        [customer_message]
    )[0]

    # Retrieve historical example
    similarity, historical_case = retrieve_case(
        customer_message
    )

    # Decide escalation
    escalate, reason = decide_escalation(
        intent,
        similarity,
        customer_message
    )

    # Create response
    if escalate:

        response = (
            "This issue needs further investigation. "
            "I recommend escalating it to a human support agent."
        )

    else:

        response = historical_case["amazon_reply"]

    return {
        "customer_message": customer_message,
        "intent": intent,
        "similarity": round(similarity, 4),
        "should_escalate": "yes" if escalate else "no",
        "escalation_reason": reason,
        "draft_reply": response,
        "historical_customer_message":
            historical_case["customer_message"]
    }


# ============================================================
# 7. TEST THE AGENT
# ============================================================

if __name__ == "__main__":

    test_messages = [

        "My package was supposed to arrive yesterday "
        "but I still haven't received it.",

        "Someone used my card without my permission.",

        "My Echo Dot is not connecting to Alexa.",

        "I want to return the product and get a refund."
    ]

    print("\n")
    print("=" * 60)
    print("SUPPORT AGENT TEST")
    print("=" * 60)

    for message in test_messages:

        result = support_agent(message)

        print("\nCustomer:")
        print(result["customer_message"])

        print("\nPredicted intent:")
        print(result["intent"])

        print("\nSimilarity:")
        print(result["similarity"])

        print("\nEscalate:")
        print(result["should_escalate"])

        print("\nReason:")
        print(result["escalation_reason"])

        print("\nDraft reply:")
        print(result["draft_reply"])

        print("\nHistorical example:")
        print(result["historical_customer_message"])

        print("\n" + "-" * 60)