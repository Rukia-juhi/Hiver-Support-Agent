import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

golden = pd.read_csv(
    "data/golden_set.csv",
    encoding="latin1"
)

predictions = pd.read_csv(
    "results/final_intent_predictions.csv",
    encoding="utf-8"
)

print("Golden test rows:", len(golden))
print("Prediction rows:", len(predictions))


# ============================================================
# 2. COMBINE DATA
# ============================================================

results = golden.copy()

results["predicted_intent"] = predictions["predicted_intent"]


# ============================================================
# 3. IMPROVED ESCALATION RULES
# ============================================================

def decide_escalation(row):

    intent = str(row["predicted_intent"]).strip().lower()
    message = str(row["customer_message"]).lower()

    # --------------------------------------------------------
    # RULE 1: ACCOUNT / SECURITY
    # --------------------------------------------------------

    security_words = [
        "hacked",
        "phishing",
        "fraud",
        "scam",
        "stolen",
        "unauthorized",
        "unauthorised",
        "suspicious",
        "account locked",
        "account hacked",
        "can't sign in",
        "cannot sign in",
        "password",
        "login",
        "log in",
        "chargeback",
        "identity theft"
    ]

    if any(word in message for word in security_words):
        return "yes"


    # --------------------------------------------------------
    # RULE 2: SELLER / MARKETPLACE DISPUTES
    # --------------------------------------------------------

    seller_words = [
        "seller",
        "third party",
        "third-party",
        "merchant",
        "a-to-z",
        "a to z guarantee",
        "marketplace seller"
    ]

    if intent == "seller_issue":
        return "yes"

    if any(word in message for word in seller_words):
        return "yes"


    # --------------------------------------------------------
    # RULE 3: EXPLICIT REQUEST FOR HUMAN HELP
    # --------------------------------------------------------

    human_help_words = [
        "speak to someone",
        "talk to someone",
        "talk to a person",
        "speak to a person",
        "human",
        "agent",
        "representative",
        "call me",
        "please call",
        "contact me",
        "customer service",
        "customer care",
        "helpline",
        "supervisor"
    ]

    if any(word in message for word in human_help_words):
        return "yes"


    # --------------------------------------------------------
    # RULE 4: UNRESOLVED / REPEATED PROBLEM
    # --------------------------------------------------------

    unresolved_words = [
        "still",
        "again",
        "yet",
        "more than",
        "already contacted",
        "already contacted them",
        "contacted customer service",
        "called customer care",
        "multiple times",
        "repeatedly",
        "nobody",
        "no one",
        "not resolved",
        "hasn't been resolved",
        "have not received",
        "didn't receive",
        "did not receive",
        "nothing happened",
        "ignored",
        "ignoring",
        "waiting",
        "weeks",
        "days"
    ]

    if any(word in message for word in unresolved_words):
        return "yes"


    # --------------------------------------------------------
    # RULE 5: DELIVERY PROBLEMS
    # --------------------------------------------------------

    delivery_escalation_words = [
        "lost",
        "delivered to someone else",
        "wrong person",
        "wrong address",
        "refusing delivery",
        "refused delivery",
        "lied",
        "courier",
        "failed delivery",
        "missed delivery",
        "late",
        "delayed",
        "past the delivery date",
        "overdue"
    ]

    if intent == "delivery_issue":
        if any(word in message for word in delivery_escalation_words):
            return "yes"


    # --------------------------------------------------------
    # RULE 6: REFUND / RETURN DISPUTES
    # --------------------------------------------------------

    refund_escalation_words = [
        "where is my refund",
        "when will i get my refund",
        "still waiting for refund",
        "refund hasn't",
        "refund has not",
        "money back",
        "money missing",
        "only got",
        "partial refund",
        "restocking fee",
        "charged a fee"
    ]

    if intent == "return_refund":
        if any(word in message for word in refund_escalation_words):
            return "yes"


    # --------------------------------------------------------
    # RULE 7: ORDER PROBLEMS
    # --------------------------------------------------------

    order_escalation_words = [
        "cancel",
        "can't cancel",
        "cannot cancel",
        "preorder",
        "pre-order",
        "out of stock",
        "order keeps",
        "order delayed",
        "order hasn't",
        "order has not",
        "order missing",
        "no confirmation"
    ]

    if intent == "order_issue":
        if any(word in message for word in order_escalation_words):
            return "yes"


    # --------------------------------------------------------
    # RULE 8: STRONG COMPLAINTS
    # --------------------------------------------------------

    complaint_words = [
        "unacceptable",
        "shameful",
        "horrible",
        "terrible",
        "worst",
        "disgusting",
        "cheating",
        "scamming",
        "ridiculous",
        "angry",
        "furious",
        "unacceptable",
        "regretting",
        "lose customers"
    ]

    if any(word in message for word in complaint_words):
        return "yes"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "no"


# ============================================================
# 4. APPLY ESCALATION RULES
# ============================================================

results["predicted_escalation"] = results.apply(
    decide_escalation,
    axis=1
)


# ============================================================
# 5. NORMALIZE LABELS
# ============================================================

results["should_escalate"] = (
    results["should_escalate"]
    .astype(str)
    .str.strip()
    .str.lower()
)

results["predicted_escalation"] = (
    results["predicted_escalation"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 6. METRICS
# ============================================================

accuracy = accuracy_score(
    results["should_escalate"],
    results["predicted_escalation"]
)

print("\n" + "=" * 70)
print("IMPROVED ESCALATION RESULTS")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        results["should_escalate"],
        results["predicted_escalation"],
        labels=["no", "yes"],
        zero_division=0
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    results["should_escalate"],
    results["predicted_escalation"],
    labels=["no", "yes"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual no", "Actual yes"],
    columns=["Predicted no", "Predicted yes"]
)

print("\nConfusion Matrix:")
print(cm_df)


# ============================================================
# 8. DISTRIBUTION
# ============================================================

print("\nActual escalation distribution:")
print(results["should_escalate"].value_counts())

print("\nPredicted escalation distribution:")
print(results["predicted_escalation"].value_counts())


# ============================================================
# 9. ERROR ANALYSIS
# ============================================================

results["correct_escalation"] = (
    results["should_escalate"]
    == results["predicted_escalation"]
)

errors = results[
    results["correct_escalation"] == False
]

print("\n" + "=" * 70)
print("ESCALATION ERRORS")
print("=" * 70)

print("\nTotal errors:", len(errors))

print("\nFirst 20 errors:")

print(
    errors[
        [
            "customer_message",
            "intent",
            "predicted_intent",
            "should_escalate",
            "predicted_escalation",
            "escalation_reason"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 10. SAVE
# ============================================================

results.to_csv(
    "results/final_escalation_results.csv",
    index=False
)

print("\nSaved:")
print("results/final_escalation_results.csv")