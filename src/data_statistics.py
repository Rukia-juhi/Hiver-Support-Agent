#Check the dataset language and message lengths

import pandas as pd

# Load AmazonHelp conversations
df = pd.read_csv("data/amazon_conversations.csv")

print("Total conversations:", len(df))

# Message lengths
df["message_length"] = df["customer_message"].str.len()

print("\nCustomer message length statistics:")
print(df["message_length"].describe())

# Very short messages
short_messages = df[df["message_length"] < 20]

print("\nMessages shorter than 20 characters:")
print(len(short_messages))

print("\nExamples of short messages:")
print(
    short_messages["customer_message"]
    .head(20)
    .to_string(index=False)
)

# Inbound/outbound isn't needed here because we already have
# customer -> AmazonHelp conversations.

# Save statistics-friendly sample
sample = df.sample(
    n=min(500, len(df)),
    random_state=42
)

sample.to_csv(
    "data/amazon_sample_500.csv",
    index=False
)

print("\nSaved random sample:")
print("data/amazon_sample_500.csv")