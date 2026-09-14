import pandas as pd

df = pd.read_csv("data/amazon_training_data.csv")


def classify_message(text):
    text = str(text).lower()

    # -----------------------------
    # 1. Account / Security
    # -----------------------------
    if any(word in text for word in [
        "hacked", "phishing", "fraud", "unauthorized",
        "stolen card", "suspicious email", "scam email",
        "account locked", "account on hold", "can't sign in",
        "cannot sign in", "password", "login", "log in"
    ]):
        return "account_security"

    # -----------------------------
    # 2. Seller issues
    # -----------------------------
    if any(word in text for word in [
        "seller", "third party", "third-party",
        "a-to-z", "a to z guarantee",
        "marketplace seller", "merchant"
    ]):
        return "seller_issue"

    # -----------------------------
    # 3. Delivery issues
    # -----------------------------
    if any(word in text for word in [
        "package", "parcel", "delivery", "delivered",
        "delivery date", "tracking", "courier",
        "shipment", "shipping", "arrive", "arrived",
        "not received", "didn't receive",
        "hasn't arrived", "late", "lost package",
        "where is my package"
    ]):
        return "delivery_issue"

    # -----------------------------
    # 4. Returns / Refunds
    # -----------------------------
    if any(word in text for word in [
        "refund", "return", "returned",
        "replacement", "replace", "money back",
        "wrong item", "damaged item",
        "defective", "broken item"
    ]):
        return "return_refund"

    # -----------------------------
    # 5. Payment / Billing
    # -----------------------------
    if any(word in text for word in [
        "charged", "charge", "payment",
        "billing", "credit card", "debit card",
        "gift card", "amazon pay", "cashback",
        "subscription fee", "membership fee",
        "money was taken", "money deducted"
    ]):
        return "payment_billing"

    # -----------------------------
    # 6. Product / Device
    # -----------------------------
    if any(word in text for word in [
        "kindle", "echo", "alexa", "fire tv",
        "firestick", "fire tablet", "amazon device",
        "device", "headphones", "speaker"
    ]):
        return "product_device_issue"

    # -----------------------------
    # 7. Digital services
    # -----------------------------
    if any(word in text for word in [
        "prime video", "amazon video",
        "streaming", "prime music",
        "music unlimited", "prime now",
        "app", "digital service", "subtitles",
        "video not working", "music not working"
    ]):
        return "digital_service_issue"

    # -----------------------------
    # 8. Order issues
    # -----------------------------
    if any(word in text for word in [
        "order", "ordered", "preorder",
        "pre-order", "cancel my order",
        "order status", "order confirmation",
        "order number", "cancel order"
    ]):
        return "order_issue"

    # -----------------------------
    # 9. Information / Feedback
    # -----------------------------
    if any(word in text for word in [
        "how much", "price", "available",
        "availability", "discount",
        "promotion", "offer", "do you sell",
        "information", "when will", "can i buy",
        "ship to", "shipping to"
    ]):
        return "information_feedback"

    # -----------------------------
    # 10. General support
    # -----------------------------
    return "general_support"


df["weak_intent"] = df["customer_message"].apply(classify_message)

print("\nImproved weak-label distribution:")
print(df["weak_intent"].value_counts())

df.to_csv(
    "data/amazon_weak_labeled.csv",
    index=False
)

print("\nSaved:")
print("data/amazon_weak_labeled.csv")
print("Rows:", len(df))