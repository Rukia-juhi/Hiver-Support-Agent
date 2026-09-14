import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load historical conversations
df = pd.read_csv("data/amazon_training_data.csv")

print("Loading historical conversations...")
print("Number of conversations:", len(df))


# Build TF-IDF representation
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000
)

print("Building TF-IDF index...")

tfidf_matrix = vectorizer.fit_transform(
    df["customer_message"].fillna("")
)

print("Index created successfully!")


def find_similar_messages(
    customer_message,
    top_k=3
):
    """
    Find the most similar historical customer messages.
    """

    query_vector = vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Get top results
    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "similarity": round(
                float(similarities[index]),
                4
            ),
            "customer_message":
                df.iloc[index]["customer_message"],
            "amazon_reply":
                df.iloc[index]["amazon_reply"]
        })

    return results


# Test the retriever
test_message = (
    "My package was supposed to arrive yesterday "
    "but I still haven't received it."
)

print("\nTest customer message:")
print(test_message)

print("\nMost similar historical conversations:")

results = find_similar_messages(
    test_message,
    top_k=3
)

for i, result in enumerate(results, start=1):

    print("\nResult", i)
    print("Similarity:", result["similarity"])

    print("Customer:")
    print(result["customer_message"])

    print("AmazonHelp reply:")
    print(result["amazon_reply"])