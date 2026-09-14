import pandas as pd

# Load the 500-message sample
df = pd.read_csv("data/amazon_sample_500.csv")

# Take 200 examples for manual labeling
golden = df.sample(
    n=200,
    random_state=42
).copy()

# Add an empty label column
golden["intent"] = ""

# Add an empty escalation label
golden["should_escalate"] = ""

# Add an empty reason column
golden["escalation_reason"] = ""

# Keep only the columns we need for labeling
golden = golden[
    [
        "customer_tweet_id",
        "customer_message",
        "amazon_reply",
        "intent",
        "should_escalate",
        "escalation_reason"
    ]
]

# Save the golden set
golden.to_csv(
    "data/golden_set.csv",
    index=False
)

print("Golden set created successfully!")
print("Number of examples:", len(golden))
print("Saved to: data/golden_set.csv")