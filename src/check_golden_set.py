import pandas as pd

# Load the labeled golden set
df = pd.read_csv("data/golden_set.csv")

print("Total rows:", len(df))

print("\nMissing values:")
print(df[
    ["intent", "should_escalate", "escalation_reason"]
].isnull().sum())

print("\nIntent distribution:")
print(df["intent"].value_counts())

print("\nEscalation distribution:")
print(df["should_escalate"].value_counts())

print("\nFirst 10 labeled examples:")
print(
    df[
        [
            "customer_message",
            "intent",
            "should_escalate",
            "escalation_reason"
        ]
    ].head(10).to_string(index=False)
)