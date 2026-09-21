import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading AmazonHelp conversations...")

df = pd.read_csv(
    "data/amazonhelp_conversations.csv"
)

print("Total conversations:", len(df))


# ============================================================
# 2. REMOVE THE 200 GOLDEN EVALUATION EXAMPLES
# ============================================================

golden = pd.read_csv(
    "evaluation/golden_set_labeled.csv",
    dtype=str
)

golden_ids = set(
    golden["tweet_id"].astype(str)
)

df["tweet_id"] = df["tweet_id"].astype(str)

df = df[
    ~df["tweet_id"].isin(golden_ids)
].copy()

print("Training candidates:", len(df))


# ============================================================
# 3. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


df["clean_text"] = df["text"].fillna("").apply(clean_text)


# ============================================================
# 4. WEAK LABELING RULES
# ============================================================

def assign_intent(text):

    text = text.lower()

    # ----------------------------------------
    # Device
    # ----------------------------------------

    if any(word in text for word in [
        "kindle",
        "fire tv",
        "firetv",
        "echo",
        "alexa",
        "device",
        "tablet"
    ]):
        return "device_technical"


    # ----------------------------------------
    # Prime
    # ----------------------------------------

    if "prime" in text:
        return "prime_membership"


    # ----------------------------------------
    # Return
    # ----------------------------------------

    if any(word in text for word in [
        "return",
        "returned",
        "returning",
        "return label",
        "return package"
    ]):
        return "return_issue"


    # ----------------------------------------
    # Refund / billing
    # ----------------------------------------

    if any(word in text for word in [
        "refund",
        "refunded",
        "charged",
        "charge",
        "billing",
        "money back"
    ]):
        return "refund_billing"


    # ----------------------------------------
    # Payment
    # ----------------------------------------

    if any(word in text for word in [
        "payment",
        "paid",
        "wallet",
        "amazon payments",
        "payment account"
    ]):
        return "payment_issue"


    # ----------------------------------------
    # Account
    # ----------------------------------------

    if any(word in text for word in [
        "account",
        "login",
        "log in",
        "password",
        "sign in",
        "email address"
    ]):
        return "account_issue"


    # ----------------------------------------
    # Delivery
    # ----------------------------------------

    if any(word in text for word in [
        "delivery",
        "delivered",
        "package",
        "shipping",
        "shipment",
        "courier",
        "tracking",
        "arrive",
        "arrived"
    ]):
        return "delivery_issue"


    # ----------------------------------------
    # Order
    # ----------------------------------------

    if any(word in text for word in [
        "order",
        "ordered",
        "pre-order",
        "preorder",
        "claim code"
    ]):
        return "order_issue"


    # ----------------------------------------
    # General complaint
    # ----------------------------------------

    if any(word in text for word in [
        "complaint",
        "terrible",
        "worst",
        "unacceptable",
        "disappointed",
        "lost a customer",
        "angry"
    ]):
        return "general_complaint"


    # No confident label
    return None


df["intent"] = df["clean_text"].apply(assign_intent)


# ============================================================
# 5. REMOVE UNKNOWN EXAMPLES
# ============================================================

df = df[
    df["intent"].notna()
].copy()

print(
    "Weakly labelled training examples:",
    len(df)
)


# ============================================================
# 6. SHOW CLASS DISTRIBUTION
# ============================================================

print("\nIntent distribution:")

print(
    df["intent"].value_counts()
)


# ============================================================
# 7. LIMIT LARGE CLASSES
# ============================================================

MAX_PER_CLASS = 10000

balanced_parts = []

for intent, group in df.groupby("intent"):

    if len(group) > MAX_PER_CLASS:

        group = group.sample(
            n=MAX_PER_CLASS,
            random_state=42
        )

    balanced_parts.append(group)


df = pd.concat(
    balanced_parts,
    ignore_index=True
)

print(
    "\nTraining examples after balancing:",
    len(df)
)


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["intent"],
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)


# ============================================================
# 9. BUILD MODEL
# ============================================================

model = Pipeline([
    
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining intent classifier...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# ============================================================
# 11. VALIDATE
# ============================================================

predictions = model.predict(
    X_test
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "experiments/intent_classifier.joblib"
)

print(
    "\nModel saved to:"
)

print(
    "experiments/intent_classifier.joblib"
)