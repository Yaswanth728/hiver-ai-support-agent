import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD GOLDEN DATA
# ============================================================

print("Loading human-labelled golden set...")

df = pd.read_csv(
    "evaluation/golden_set_labeled.csv",
    dtype=str
)

df["text"] = df["text"].fillna("")
df["intent"] = df["intent"].fillna("")

print("Golden examples:", len(df))


# ============================================================
# 2. TF-IDF REPRESENTATION
# ============================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    df["text"]
)

print("TF-IDF matrix created.")


# ============================================================
# 3. RETRIEVAL FUNCTION
# ============================================================

def predict_intent(
    message,
    exclude_index=None,
    top_k=5
):

    query_vector = vectorizer.transform(
        [message]
    )

    similarities = cosine_similarity(
        query_vector,
        X
    )[0]

    # Exclude the same example during evaluation
    if exclude_index is not None:
        similarities[exclude_index] = -1

    # Get top similar examples
    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    # Collect candidate intents
    candidates = []

    for idx in top_indices:

        candidates.append({
            "index": idx,
            "intent": df.iloc[idx]["intent"],
            "similarity": similarities[idx],
            "text": df.iloc[idx]["text"]
        })

    # --------------------------------------------------------
    # Majority vote weighted by similarity
    # --------------------------------------------------------

    intent_scores = {}

    for item in candidates:

        intent = item["intent"]
        similarity = item["similarity"]

        if similarity > 0:
            intent_scores[intent] = (
                intent_scores.get(intent, 0)
                + similarity
            )

    if not intent_scores:
        predicted_intent = "general_complaint"
    else:
        predicted_intent = max(
            intent_scores,
            key=intent_scores.get
        )

    return predicted_intent, candidates


# ============================================================
# 4. LEAVE-ONE-OUT EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("LEAVE-ONE-OUT GOLDEN SET EVALUATION")
print("=" * 70)

actual = []
predicted = []

for i in range(len(df)):

    message = df.iloc[i]["text"]
    true_intent = df.iloc[i]["intent"]

    prediction, _ = predict_intent(
        message,
        exclude_index=i,
        top_k=5
    )

    actual.append(true_intent)
    predicted.append(prediction)


# ============================================================
# 5. RESULTS
# ============================================================

accuracy = accuracy_score(
    actual,
    predicted
)

print(
    f"\nRetrieval Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        actual,
        predicted,
        zero_division=0
    )
)


# ============================================================
# 6. SAVE PREDICTIONS
# ============================================================

df["retrieval_predicted_intent"] = predicted

df["retrieval_correct"] = (
    df["intent"] ==
    df["retrieval_predicted_intent"]
)

df.to_csv(
    "evaluation/retrieval_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nPredictions saved to:"
)

print(
    "evaluation/retrieval_predictions.csv"
)


# ============================================================
# 7. SAVE TF-IDF DATA
# ============================================================

joblib.dump(
    vectorizer,
    "experiments/retrieval_vectorizer.joblib"
)

print(
    "\nVectorizer saved to:"
)

print(
    "experiments/retrieval_vectorizer.joblib"
)


# ============================================================
# 8. INTERACTIVE TEST
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE INTENT TEST")
print("=" * 70)

print(
    "\nType a customer message to test the retrieval classifier."
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

    prediction, matches = predict_intent(
        message,
        top_k=5
    )

    print(
        f"\nPredicted intent: {prediction}"
    )

    print(
        "\nTop similar historical examples:"
    )

    for rank, item in enumerate(
        matches,
        start=1
    ):

        print(
            f"\n{rank}. "
            f"[{item['similarity']:.3f}] "
            f"{item['intent']}"
        )

        print(
            f"   {item['text'][:250]}"
        )