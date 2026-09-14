import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. LOAD DATA
# ============================================================

golden = pd.read_csv("data/golden_set.csv", encoding="latin1")
historical = pd.read_csv("data/amazon_training_data.csv", encoding="utf-8")

print("Golden test rows:", len(golden))
print("Historical training rows:", len(historical))

# ============================================================
# 2. REMOVE GOLDEN-SET LEAKAGE
# ============================================================

# Remove by tweet ID
golden_ids = set(golden["customer_tweet_id"].astype(str))

historical["customer_tweet_id"] = (
    historical["customer_tweet_id"].astype(str)
)

historical = historical[
    ~historical["customer_tweet_id"].isin(golden_ids)
].copy()

print(
    "After removing golden IDs:",
    len(historical)
)

# Remove exact customer-message duplicates
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

before_text_removal = len(historical)

historical = historical[
    ~historical["message_clean"].isin(golden_messages)
].copy()

removed_text_duplicates = (
    before_text_removal - len(historical)
)

print(
    "Exact golden-message duplicates removed:",
    removed_text_duplicates
)

print(
    "Final historical retrieval corpus:",
    len(historical)
)

# ============================================================
# 3. PREPARE TEXT
# ============================================================

train_text = (
    historical["customer_message"]
    .fillna("")
    .astype(str)
)

test_text = (
    golden["customer_message"]
    .fillna("")
    .astype(str)
)

# ============================================================
# 4. BUILD TF-IDF INDEX
# ============================================================

print("\nBuilding TF-IDF retrieval index...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_features=50000,
    sublinear_tf=True
)

train_vectors = vectorizer.fit_transform(train_text)
test_vectors = vectorizer.transform(test_text)

print("Training matrix:", train_vectors.shape)
print("Test matrix:", test_vectors.shape)

# ============================================================
# 5. RETRIEVE HISTORICAL RESPONSES
# ============================================================

print("\nRetrieving historical responses...")

results = []

for i in range(len(golden)):

    query_vector = test_vectors[i]

    similarities = cosine_similarity(
        query_vector,
        train_vectors
    ).flatten()

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    best_row = historical.iloc[best_index]

    results.append({
        "customer_tweet_id":
            golden.iloc[i]["customer_tweet_id"],

        "customer_message":
            golden.iloc[i]["customer_message"],

        "gold_intent":
            golden.iloc[i]["intent"],

        "gold_reply":
            golden.iloc[i]["amazon_reply"],

        "retrieved_message":
            best_row["customer_message"],

        "retrieved_reply":
            best_row["amazon_reply"],

        "similarity":
            best_score
    })

results_df = pd.DataFrame(results)

# ============================================================
# 6. RETRIEVAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE-FREE RETRIEVAL RESULTS")
print("=" * 70)

print(
    f"\nAverage similarity: "
    f"{results_df['similarity'].mean():.4f}"
)

print(
    f"Median similarity: "
    f"{results_df['similarity'].median():.4f}"
)

print(
    f"Minimum similarity: "
    f"{results_df['similarity'].min():.4f}"
)

print(
    f"Maximum similarity: "
    f"{results_df['similarity'].max():.4f}"
)

print("\nSimilarity ranges:")

print(
    pd.cut(
        results_df["similarity"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True
    )
    .value_counts()
    .sort_index()
)

# ============================================================
# 7. TOP 10
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 RETRIEVAL EXAMPLES")
print("=" * 70)

top10 = (
    results_df
    .sort_values("similarity", ascending=False)
    .head(10)
)

for _, row in top10.iterrows():

    print("\n" + "-" * 70)

    print("CUSTOMER:")
    print(row["customer_message"])

    print("\nRETRIEVED CUSTOMER:")
    print(row["retrieved_message"])

    print(
        f"\nSIMILARITY: "
        f"{row['similarity']:.4f}"
    )

    print("\nRETRIEVED AMAZON REPLY:")
    print(row["retrieved_reply"])

# ============================================================
# 8. LOW-SIMILARITY EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("LOW-SIMILARITY RETRIEVALS")
print("=" * 70)

low10 = (
    results_df
    .sort_values("similarity", ascending=True)
    .head(10)
)

for _, row in low10.iterrows():

    print("\n" + "-" * 70)

    print("CUSTOMER:")
    print(row["customer_message"])

    print(
        f"\nSIMILARITY: "
        f"{row['similarity']:.4f}"
    )

    print("\nRETRIEVED CUSTOMER:")
    print(row["retrieved_message"])

    print("\nRETRIEVED REPLY:")
    print(row["retrieved_reply"])

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "results/retrieval_evaluation.csv",
    index=False
)

print("\nSaved:")
print("results/retrieval_evaluation.csv")