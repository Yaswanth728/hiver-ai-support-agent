import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD RETRIEVAL SYSTEM
# ============================================================

print("Loading retrieval system...")

vectorizer = joblib.load(
    "experiments/response_vectorizer.joblib"
)

tfidf_matrix = joblib.load(
    "experiments/response_tfidf_matrix.joblib"
)

historical_df = pd.read_pickle(
    "experiments/retrieval_conversations.pkl"
)

print(
    f"Historical conversations: {len(historical_df)}"
)


# ============================================================
# 2. LOAD GOLDEN SET
# ============================================================

golden_file = (
    "evaluation/golden_set_labeled.csv"
)

golden_df = pd.read_csv(
    golden_file,
    dtype=str
)

golden_df["text"] = (
    golden_df["text"]
    .fillna("")
)


print(
    f"Golden-set examples: {len(golden_df)}"
)


# ============================================================
# 3. RETRIEVE HISTORICAL EXAMPLES
# ============================================================

def retrieve(
    message,
    top_k=5
):

    query_vector = vectorizer.transform(
        [message.lower()]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            "similarity": float(
                similarities[idx]
            ),
            "customer": historical_df.iloc[idx]["text"],
            "response": historical_df.iloc[idx]["response_text"]
        })

    return results


# ============================================================
# 4. EVALUATE RETRIEVAL
# ============================================================

print("\n" + "=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)

results = []

similarities = []


for i, row in golden_df.iterrows():

    message = row["text"]

    retrieved = retrieve(
        message,
        top_k=5
    )

    if retrieved:

        best = retrieved[0]

        similarity = best["similarity"]

        similarities.append(
            similarity
        )

        results.append({
            "tweet_id": row.get(
                "tweet_id",
                ""
            ),
            "text": message,
            "expected_intent": row.get(
                "intent",
                ""
            ),
            "best_similarity": similarity,
            "retrieved_customer": best[
                "customer"
            ],
            "retrieved_response": best[
                "response"
            ]
        })


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

results_df = pd.DataFrame(
    results
)

average_similarity = (
    sum(similarities) /
    len(similarities)
    if similarities
    else 0
)

thresholds = [
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]


print(
    f"\nAverage top-1 similarity: "
    f"{average_similarity:.3f}"
)

print("\nRetrieval coverage:")

for threshold in thresholds:

    count = sum(
        s >= threshold
        for s in similarities
    )

    percentage = (
        count /
        len(similarities) *
        100
        if similarities
        else 0
    )

    print(
        f"Similarity >= {threshold:.2f}: "
        f"{count}/{len(similarities)} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

output_file = (
    "evaluation/retrieval_evaluation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(
    f"\nResults saved to: {output_file}"
)


# ============================================================
# 7. SHOW SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE RETRIEVAL RESULTS")
print("=" * 70)


for i, result in enumerate(
    results[:10],
    start=1
):

    print(
        f"\n{i}. Similarity: "
        f"{result['best_similarity']:.3f}"
    )

    print(
        "Customer:"
    )

    print(
        result["text"][:300]
    )

    print(
        "\nRetrieved historical customer:"
    )

    print(
        result["retrieved_customer"][:300]
    )

    print(
        "\nRetrieved AmazonHelp response:"
    )

    print(
        result["retrieved_response"][:400]
    )

    print(
        "\n" + "-" * 70
    )


print(
    "\nRETRIEVAL EVALUATION COMPLETE!"
)