import pandas as pd

# Load AmazonHelp conversations
df = pd.read_csv("data/amazon_conversations.csv")

print("Total conversations:", len(df))

print("\nMissing values:")
print(df.isnull().sum())

print("\nSample customer messages:\n")

for i, message in enumerate(df["customer_message"].head(30), start=1):
    print(f"{i}. {message}")

print("\n--------------------------------")
print("Sample AmazonHelp replies:")
print("--------------------------------\n")

for i, reply in enumerate(df["amazon_reply"].head(10), start=1):
    print(f"{i}. {reply}")