import pandas as pd

# Load dataset
df = pd.read_csv("data/twcs.csv")

# Keep AmazonHelp tweets
amazon = df[df["author_id"] == "AmazonHelp"].copy()

print("AmazonHelp tweets:", len(amazon))

# Create a lookup table using tweet_id
tweets = df.set_index("tweet_id")

conversations = []

for _, row in amazon.iterrows():

    # Find the customer tweet that AmazonHelp replied to
    parent_id = row["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    if parent_id not in tweets.index:
        continue

    # Get the customer tweet
    customer = tweets.loc[parent_id]

    # Make sure the parent is a customer message
    if customer["inbound"] != True:
        continue

    conversations.append({
        "customer_tweet_id": parent_id,
        "amazon_tweet_id": row["tweet_id"],
        "customer_message": customer["text"],
        "amazon_reply": row["text"]
    })

# Convert to DataFrame
conversations_df = pd.DataFrame(conversations)

print("\nCustomer → AmazonHelp conversations:",
      len(conversations_df))

print("\nFirst 10 conversations:\n")

print(
    conversations_df[
        [
            "customer_tweet_id",
            "amazon_tweet_id",
            "customer_message",
            "amazon_reply"
        ]
    ].head(10).to_string(index=False)
)

# Save conversations
conversations_df.to_csv(
    "data/amazon_conversations.csv",
    index=False
)

print("\nSaved successfully:")
print("data/amazon_conversations.csv")