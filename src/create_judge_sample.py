import pandas as pd

# Load retrieval results
df = pd.read_csv(
    "results/retrieval_evaluation.csv",
    encoding="latin1"
)

print("Total retrieval results:", len(df))

# Create similarity groups
df["similarity_bin"] = pd.cut(
    df["similarity"],
    bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
    labels=[
        "very_low",
        "low",
        "medium",
        "high",
        "very_high"
    ]
)

samples = []

# Try to take 10 from each similarity group
for category in [
    "very_low",
    "low",
    "medium",
    "high",
    "very_high"
]:

    subset = df[df["similarity_bin"] == category]

    if len(subset) == 0:
        continue

    n = min(10, len(subset))

    samples.append(
        subset.sample(
            n=n,
            random_state=42
        )
    )

judge_sample = pd.concat(
    samples,
    ignore_index=True
)

# Remove duplicates just in case
judge_sample = judge_sample.drop_duplicates(
    subset=["customer_tweet_id"]
)

# Fill remaining rows randomly from examples
# not already selected
remaining_needed = 50 - len(judge_sample)

if remaining_needed > 0:

    selected_ids = set(
        judge_sample["customer_tweet_id"]
    )

    remaining = df[
        ~df["customer_tweet_id"].isin(selected_ids)
    ]

    extra = remaining.sample(
        n=min(remaining_needed, len(remaining)),
        random_state=123
    )

    judge_sample = pd.concat(
        [judge_sample, extra],
        ignore_index=True
    )

# Make sure exactly 50 if possible
judge_sample = judge_sample.head(50)

# Add human evaluation columns
judge_sample["human_relevance"] = ""
judge_sample["human_helpfulness"] = ""
judge_sample["human_groundedness"] = ""
judge_sample["human_tone"] = ""
judge_sample["human_total"] = ""
judge_sample["human_pass"] = ""
judge_sample["human_notes"] = ""

# Keep required columns
judge_sample = judge_sample[
    [
        "customer_tweet_id",
        "customer_message",
        "gold_intent",
        "gold_reply",
        "retrieved_message",
        "retrieved_reply",
        "similarity",
        "human_relevance",
        "human_helpfulness",
        "human_groundedness",
        "human_tone",
        "human_total",
        "human_pass",
        "human_notes"
    ]
]

# Save
judge_sample.to_csv(
    "data/judge_sample.csv",
    index=False
)

print("\nJudge sample created!")
print("Rows:", len(judge_sample))
print("Saved:")
print("data/judge_sample.csv")

print("\nSimilarity distribution:")
print(
    pd.cut(
        judge_sample["similarity"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01]
    )
    .value_counts()
    .sort_index()
)