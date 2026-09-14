import pandas as pd

# Load dataset
df = pd.read_csv("data/twcs.csv")

# Select AmazonHelp
amazon_df = df[df["author_id"] == "AmazonHelp"].copy()

print("AmazonHelp tweets:", len(amazon_df))

print("\nFirst 10 AmazonHelp tweets:")
print(
    amazon_df[
        ["tweet_id", "author_id", "inbound", "text"]
    ].head(10).to_string(index=False)
)