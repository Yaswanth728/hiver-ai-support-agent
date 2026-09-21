import pandas as pd

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/twcs.csv")

print("Dataset loaded successfully!")
print("Total tweets:", len(df))


# --------------------------------------------------
# 2. Convert tweet IDs to strings
# --------------------------------------------------

df["tweet_id"] = df["tweet_id"].astype(str).str.strip()

df["response_tweet_id"] = (
    df["response_tweet_id"]
    .fillna("")
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
)


# --------------------------------------------------
# 3. Select AmazonHelp tweets
# --------------------------------------------------

amazon = df[df["author_id"] == "AmazonHelp"].copy()

print("\nAmazonHelp tweets:", len(amazon))


# --------------------------------------------------
# 4. Get AmazonHelp tweet IDs
# --------------------------------------------------

amazon_tweet_ids = set(amazon["tweet_id"])


# --------------------------------------------------
# 5. Find customer tweets answered by AmazonHelp
# --------------------------------------------------

customer_messages = df[
    (df["inbound"] == True) &
    (df["response_tweet_id"].isin(amazon_tweet_ids))
].copy()

print("\nCustomer messages answered by AmazonHelp:")
print(len(customer_messages))


# --------------------------------------------------
# 6. Connect customer tweets to responses
# --------------------------------------------------

amazon_responses = amazon[
    ["tweet_id", "text"]
].rename(
    columns={
        "tweet_id": "response_tweet_id",
        "text": "response_text"
    }
)

conversations = customer_messages[
    ["tweet_id", "author_id", "text", "response_tweet_id"]
].merge(
    amazon_responses,
    on="response_tweet_id",
    how="inner"
)


# --------------------------------------------------
# 7. Display examples
# --------------------------------------------------

print("\nCustomer → AmazonHelp examples:\n")

for _, row in conversations.head(10).iterrows():

    print("=" * 80)

    print("CUSTOMER:")
    print(row["text"])

    print("\nAMAZONHELP:")
    print(row["response_text"])

    print()


# --------------------------------------------------
# 8. Save conversations
# --------------------------------------------------

conversations.to_csv(
    "data/amazonhelp_conversations.csv",
    index=False
)

print("=" * 80)
print("Saved conversation dataset!")
print("File: data/amazonhelp_conversations.csv")