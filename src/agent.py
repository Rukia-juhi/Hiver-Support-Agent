import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("AMAZON AI SUPPORT AGENT")
print("=" * 70)

print("\nLoading data...")

training = pd.read_csv(
    "data/amazon_weak_labeled.csv",
    encoding="utf-8"
)

golden = pd.read_csv(
    "data/golden_set.csv",
    encoding="latin1"
)

historical = pd.read_csv(
    "data/amazon_training_data.csv",
    encoding="utf-8"
)

print(f"Training rows: {len(training)}")
print(f"Historical rows: {len(historical)}")


# ============================================================
# 2. REMOVE GOLDEN SET FROM CLASSIFIER TRAINING
# ============================================================

golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

training["customer_tweet_id"] = (
    training["customer_tweet_id"].astype(str)
)

training = training[
    ~training["customer_tweet_id"].isin(golden_ids)
].copy()


# ============================================================
# 3. TRAIN INTENT CLASSIFIER
# ============================================================

print("\nTraining intent classifier...")

X_train = (
    training["customer_message"]
    .fillna("")
    .astype(str)
)

y_train = (
    training["weak_intent"]
    .astype(str)
)

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

X_train_features = features.fit_transform(X_train)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_features, y_train)

print("Intent classifier ready.")


# ============================================================
# 4. PREPARE HISTORICAL RETRIEVAL DATA
# ============================================================

print("\nBuilding historical retrieval index...")

# Remove golden-set IDs
historical["customer_tweet_id"] = (
    historical["customer_tweet_id"].astype(str)
)

historical = historical[
    ~historical["customer_tweet_id"].isin(golden_ids)
].copy()


# Remove exact golden-message duplicates
golden_messages = set(
    golden["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

historical["message_clean"] = (
    historical["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

historical = historical[
    ~historical["message_clean"].isin(golden_messages)
].copy()


retrieval_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_features=50000,
    sublinear_tf=True
)

historical_vectors = retrieval_vectorizer.fit_transform(
    historical["customer_message"]
    .fillna("")
    .astype(str)
)

print(
    f"Retrieval corpus: {len(historical)} conversations"
)


# ============================================================
# 5. ESCALATION RULES
# ============================================================

def decide_escalation(intent, message):

    message = str(message).lower()
    intent = str(intent).strip().lower()

    # --------------------------------------------------------
    # ACCOUNT / SECURITY
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
        return "yes", "Account or security issue detected."


    # --------------------------------------------------------
    # SELLER / MARKETPLACE
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
        return "yes", "Seller or marketplace dispute."

    if any(word in message for word in seller_words):
        return "yes", "Seller or marketplace dispute."


    # --------------------------------------------------------
    # REQUEST FOR HUMAN HELP
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
        return "yes", "Customer explicitly requested human assistance."


    # --------------------------------------------------------
    # UNRESOLVED / REPEATED PROBLEM
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
        return "yes", "The issue appears unresolved or repeated."


    # --------------------------------------------------------
    # DELIVERY
    # --------------------------------------------------------

    delivery_words = [
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
        if any(word in message for word in delivery_words):
            return "yes", "Unresolved or problematic delivery issue."


    # --------------------------------------------------------
    # REFUND / RETURN
    # --------------------------------------------------------

    refund_words = [
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
        if any(word in message for word in refund_words):
            return "yes", "Refund or return dispute requires attention."


    # --------------------------------------------------------
    # ORDER
    # --------------------------------------------------------

    order_words = [
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
        if any(word in message for word in order_words):
            return "yes", "Order problem may require human intervention."


    # --------------------------------------------------------
    # STRONG COMPLAINT
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
        "regretting",
        "lose customers"
    ]

    if any(word in message for word in complaint_words):
        return "yes", "Strong customer complaint detected."


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "no", "No strong escalation signal detected."


# ============================================================
# 6. PROCESS ONE CUSTOMER MESSAGE
# ============================================================

def run_agent(customer_message):

    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------

    query_features = features.transform(
        [customer_message]
    )

    predicted_intent = model.predict(
        query_features
    )[0]


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    query_vector = retrieval_vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        historical_vectors
    ).flatten()

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    best_row = historical.iloc[best_index]


    # --------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------

    escalation, reason = decide_escalation(
        predicted_intent,
        customer_message
    )


    # --------------------------------------------------------
    # DRAFT REPLY
    # --------------------------------------------------------

    historical_reply = str(
        best_row["amazon_reply"]
    )

    if escalation == "yes":

        draft_reply = (
            "I’m sorry you’re experiencing this issue. "
            "I understand your concern. Your case should "
            "be reviewed by a customer support representative "
            "who can check the specific details of your order "
            "and help resolve the issue.\n\n"
            "Reference from a similar Amazon support case:\n"
            + historical_reply
        )

    else:

        draft_reply = historical_reply


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("AI SUPPORT AGENT RESULT")
    print("=" * 70)

    print("\nCUSTOMER MESSAGE:")
    print(customer_message)

    print("\nPREDICTED INTENT:")
    print(predicted_intent)

    print("\nSIMILAR HISTORICAL CUSTOMER MESSAGE:")
    print(best_row["customer_message"])

    print(
        f"\nSIMILARITY SCORE: {best_score:.4f}"
    )

    print("\nHISTORICAL AMAZON REPLY:")
    print(historical_reply)

    print("\nDRAFT REPLY:")
    print(draft_reply)

    print("\nESCALATION DECISION:")
    print(
        "ESCALATE" if escalation == "yes"
        else "AUTO-HANDLE"
    )

    print("\nESCALATION REASON:")
    print(reason)

    print("\n" + "=" * 70)


# ============================================================
# 7. INTERACTIVE LOOP
# ============================================================

print("\nAgent is ready.")

while True:

    customer_message = input(
        "\nEnter a customer message "
        "(or type 'exit' to quit):\n> "
    ).strip()

    if customer_message.lower() == "exit":
        print("\nExiting agent.")
        break

    if not customer_message:
        print("Please enter a message.")
        continue

    run_agent(customer_message)