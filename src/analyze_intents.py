import sys
import pandas as pd

# Use UTF-8 so emojis and non-English text are handled correctly
sys.stdout.reconfigure(encoding="utf-8")

# Load conversations
df = pd.read_csv("data/amazonhelp_conversations.csv")

topics = {
    "delivery": ["delivery", "delivered", "package", "shipping"],
    "refund": ["refund", "money", "charge"],
    "order": ["order", "ordered"],
    "account": ["account", "login", "password"],
    "return": ["return", "returned"],
    "payment": ["payment", "paid"],
    "device": ["echo", "fire", "kindle", "device"],
    "prime": ["prime"],
}

print("Total conversations:", len(df))

for topic, keywords in topics.items():

    print("\n" + "=" * 80)
    print(f"TOPIC: {topic.upper()}")
    print("=" * 80)

    pattern = "|".join(keywords)

    matches = df[
        df["text"]
        .fillna("")
        .str.contains(pattern, case=False, regex=True)
    ]

    print("Matching conversations:", len(matches))

    # Print only 5 examples
    for i, message in enumerate(matches["text"].head(5), start=1):
        print(f"\n{i}. {message}")