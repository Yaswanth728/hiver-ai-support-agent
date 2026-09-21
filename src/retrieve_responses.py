import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD HISTORICAL CONVERSATIONS
# ============================================================

print("Loading AmazonHelp conversations...")

df = pd.read_csv(
    "data/amazonhelp_conversations.csv",
    dtype=str
)

df["text"] = df["text"].fillna("")
df["response_text"] = df["response_text"].fillna("")

# Remove conversations without a customer message
df = df[
    df["text"].str.strip() != ""
].copy()

print("Historical conversations:", len(df))


# ============================================================
# 2. BUILD SEARCH TEXT
# ============================================================

customer_messages = (
    df["text"]
    .str.lower()
    .str.strip()
)


# ============================================================
# 3. CREATE TF-IDF INDEX
# ============================================================

print("\nBuilding TF-IDF search index...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    customer_messages
)

print("TF-IDF index created!")
print("Matrix shape:", X.shape)


# ============================================================
# 4. RETRIEVAL FUNCTION
# ============================================================

def retrieve_responses(
    customer_message,
    top_k=5
):

    query = customer_message.lower().strip()

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        X
    )[0]

    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        results.append({
            "rank": rank,
            "similarity": float(
                similarities[idx]
            ),
            "customer_message": df.iloc[idx]["text"],
            "amazonhelp_response": df.iloc[idx]["response_text"],
            "tweet_id": df.iloc[idx]["tweet_id"]
        })

    return results


# ============================================================
# 5. SAVE SEARCH INDEX
# ============================================================

print("\nSaving retrieval index...")

joblib.dump(
    vectorizer,
    "experiments/response_vectorizer.joblib"
)

joblib.dump(
    X,
    "experiments/response_tfidf_matrix.joblib"
)

# Save the conversation records used by the index
df.to_pickle(
    "experiments/retrieval_conversations.pkl"
)

print("Retrieval index saved successfully!")


# ============================================================
# 6. INTERACTIVE TEST
# ============================================================

print("\n" + "=" * 70)
print("AMAZONHELP HISTORICAL RESPONSE RETRIEVAL")
print("=" * 70)

print(
    "\nEnter a customer message."
)

print(
    "Type 'exit' to stop."
)


while True:

    message = input(
        "\nCustomer message: "
    ).strip()

    if message.lower() == "exit":
        break

    if not message:
        continue

    results = retrieve_responses(
        message,
        top_k=5
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "TOP HISTORICAL MATCHES"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"\n{result['rank']}. "
            f"Similarity: "
            f"{result['similarity']:.3f}"
        )

        print(
            f"Tweet ID: "
            f"{result['tweet_id']}"
        )

        print(
            "\nCUSTOMER:"
        )

        print(
            result["customer_message"][
                :500
            ]
        )

        print(
            "\nAMAZONHELP:"
        )

        print(
            result["amazonhelp_response"][
                :500
            ]
        )

        print(
            "\n" + "-" * 70
        )