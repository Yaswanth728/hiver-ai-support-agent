import pandas as pd
import joblib
import re

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD RETRIEVAL COMPONENTS
# ============================================================

print("Loading retrieval system...")

vectorizer = joblib.load(
    "experiments/response_vectorizer.joblib"
)

tfidf_matrix = joblib.load(
    "experiments/response_tfidf_matrix.joblib"
)

df = pd.read_pickle(
    "experiments/retrieval_conversations.pkl"
)

print("Retrieval system loaded!")
print("Historical conversations:", len(df))


# ============================================================
# 2. CLEAN RESPONSE
# ============================================================

def clean_response(response):

    response = str(response)

    # Remove Twitter-style username at beginning
    response = re.sub(
        r"^@\w+\s*",
        "",
        response
    )

    # Remove URLs
    response = re.sub(
        r"https?://\S+",
        "",
        response
    )

    # Remove trailing agent initials such as ^AM
    response = re.sub(
        r"\s+\^[A-Z]{1,4}$",
        "",
        response
    )

    # Normalize whitespace
    response = re.sub(
        r"\s+",
        " ",
        response
    )

    return response.strip()


# ============================================================
# 3. RETRIEVE HISTORICAL RESPONSES
# ============================================================

def retrieve(
    customer_message,
    top_k=5
):

    query_vector = vectorizer.transform(
        [customer_message.lower()]
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
            "customer": df.iloc[idx]["text"],
            "response": df.iloc[idx]["response_text"],
            "tweet_id": df.iloc[idx]["tweet_id"]
        })

    return results


# ============================================================
# 4. GENERATE GROUNDED RESPONSE
# ============================================================

def generate_response(
    customer_message
):

    results = retrieve(
        customer_message,
        top_k=5
    )

    if not results:
        return {
            "response": (
                "I'm sorry you're experiencing this. "
                "Please contact customer support so we "
                "can look into the issue further."
            ),
            "evidence": []
        }

    best = results[0]

    similarity = best["similarity"]

    historical_response = clean_response(
        best["response"]
    )

    # --------------------------------------------------------
    # Low similarity = don't blindly reuse historical answer
    # --------------------------------------------------------

    if similarity < 0.25:

        response = (
            "I'm sorry you're experiencing this. "
            "We'd like to understand the issue better "
            "and help you further. Please contact our "
            "support team with the relevant details."
        )

    else:

        response = historical_response

    return {
        "response": response,
        "similarity": similarity,
        "evidence": results
    }


# ============================================================
# 5. INTERACTIVE TEST
# ============================================================

print("\n" + "=" * 70)
print("HIVER SUPPORT RESPONSE GENERATOR")
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

    result = generate_response(
        message
    )

    print(
        "\n" + "-" * 70
    )

    print("GENERATED RESPONSE")
    print("-" * 70)

    print(
        result["response"]
    )

    print(
        f"\nBest similarity: "
        f"{result['similarity']:.3f}"
    )

    print(
        "\nEVIDENCE USED"
    )

    print("-" * 70)

    for i, evidence in enumerate(
        result["evidence"],
        start=1
    ):

        print(
            f"\n{i}. Similarity: "
            f"{evidence['similarity']:.3f}"
        )

        print(
            "Historical customer:"
        )

        print(
            evidence["customer"][:300]
        )

        print(
            "\nHistorical AmazonHelp response:"
        )

        print(
            clean_response(
                evidence["response"]
            )[:500]
        )

        print(
            "\n" + "-" * 70
        )