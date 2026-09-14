import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load golden dataset
df = pd.read_csv("data/golden_set.csv")

# Remove missing labels
df = df.dropna(subset=["customer_message", "intent"])

# Find the most common intent
majority_class = df["intent"].value_counts().idxmax()
majority_count = df["intent"].value_counts().max()

# Predict the majority class for every example
y_true = df["intent"]
y_pred = [majority_class] * len(df)

accuracy = accuracy_score(y_true, y_pred)

print("=" * 60)
print("MAJORITY CLASS BASELINE")
print("=" * 60)

print("Majority class:", majority_class)
print("Examples in majority class:", majority_count)
print("Total examples:", len(df))
print(f"Baseline accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_true, y_pred, zero_division=0))