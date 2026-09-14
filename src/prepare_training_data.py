import pandas as pd


# Load AmazonHelp conversations
df = pd.read_csv("data/amazon_conversations.csv")

print("Original conversations:", len(df))

# Remove missing messages/replies
df = df.dropna(
    subset=["customer_message", "amazon_reply"]
).copy()

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"]
).copy()

# Remove very short messages
df = df[
    df["customer_message"].str.len() >= 10
].copy()

print("After cleaning:", len(df))

# Show some examples
print("\nSample training examples:")
print(
    df[
        ["customer_message", "amazon_reply"]
    ].sample(
        10,
        random_state=42
    ).to_string(index=False)
)

# Save cleaned dataset
df.to_csv(
    "data/amazon_training_data.csv",
    index=False
)

print("\nSaved:")
print("data/amazon_training_data.csv")