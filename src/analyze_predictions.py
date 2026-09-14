import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load predictions
df = pd.read_csv("results/intent_predictions.csv")

print("=" * 70)
print("INTENT CLASSIFICATION ERROR ANALYSIS")
print("=" * 70)

print("\nClassification Report:")
print(
    classification_report(
        df["intent"],
        df["predicted_intent"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
labels = sorted(df["intent"].unique())

cm = confusion_matrix(
    df["intent"],
    df["predicted_intent"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

print("\n" + "=" * 70)
print("WRONG PREDICTIONS")
print("=" * 70)

wrong = df[df["intent"] != df["predicted_intent"]]

print("Wrong predictions:", len(wrong))
print("Correct predictions:", len(df) - len(wrong))

print("\nExamples of errors:")

print(
    wrong[
        [
            "customer_message",
            "intent",
            "predicted_intent"
        ]
    ].head(30).to_string(index=False)
)