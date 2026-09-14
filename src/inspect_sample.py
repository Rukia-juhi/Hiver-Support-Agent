import pandas as pd

# Load the random sample
df = pd.read_csv("data/amazon_sample_500.csv")

print("Total sample conversations:", len(df))

print("\n" + "=" * 70)
print("CUSTOMER MESSAGES")
print("=" * 70)

for i, message in enumerate(df["customer_message"], start=1):
    print(f"\n{i}. {message}")

print("\n" + "=" * 70)
print("AMAZONHELP REPLIES")
print("=" * 70)

for i, reply in enumerate(df["amazon_reply"], start=1):
    print(f"\n{i}. {reply}")