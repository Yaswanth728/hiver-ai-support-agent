import pandas as pd
import joblib
import re

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD MODELS AND DATA
# ============================================================

print("Loading Hiver AI Support Agent...")

intent_model = joblib.load(
    "experiments/intent_classifier.joblib"
)

vectorizer = joblib.load(
    "experiments/response_vectorizer.joblib"
)

tfidf_matrix = joblib.load(
    "experiments/response_tfidf_matrix.joblib"
)

historical_df = pd.read_pickle(
    "experiments/retrieval_conversations.pkl"
)

print("Agent loaded successfully!")
print(
    f"Historical conversations available: "
    f"{len(historical_df)}"
)


# ============================================================
# 2. CLEAN HISTORICAL RESPONSE
# ============================================================

def clean_response(response):

    response = str(response)

    # Remove username at beginning
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

    # Remove HTML entities
    response = response.replace(
        "&amp;",
        "&"
    )

    # Remove Twitter agent codes such as ^AM
    response = re.sub(
        r"\s+\^[A-Z]{1,4}$",
        "",
        response
    )

    # Remove excessive whitespace
    response = re.sub(
        r"\s+",
        " ",
        response
    )

    return response.strip()


# ============================================================
# 3. PREDICT INTENT
# ============================================================

def predict_intent(message):

    prediction = intent_model.predict(
        [message]
    )[0]

    probabilities = intent_model.predict_proba(
        [message]
    )[0]

    confidence = max(
        probabilities
    )

    return prediction, confidence


# ============================================================
# 4. RETRIEVE HISTORICAL CASES
# ============================================================

def retrieve_similar_cases(
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
            "response": historical_df.iloc[idx]["response_text"],
            "tweet_id": historical_df.iloc[idx]["tweet_id"]
        })

    return results


# ============================================================
# 5. RESPONSE QUALITY CHECK
# ============================================================

def is_useful_response(response):

    response = clean_response(
        response
    )

    if not response:
        return False

    # Ignore extremely short responses
    if len(response) < 20:
        return False

    # Avoid responses that contain only generic phrases
    generic_phrases = [
        "thanks for reaching out",
        "we're here to help",
        "keep us posted",
        "that's great news",
        "thank you"
    ]

    lower_response = response.lower()

    if lower_response in generic_phrases:
        return False

    return True


# ============================================================
# 6. SELECT BEST HISTORICAL RESPONSE
# ============================================================

def select_best_response(
    results
):

    for result in results:

        response = clean_response(
            result["response"]
        )

        if is_useful_response(
            response
        ):

            return (
                response,
                result["similarity"]
            )

    return (
        None,
        0.0
    )


# ============================================================
# 7. ESCALATION POLICY
# ============================================================

def decide_handling(
    intent,
    classifier_confidence,
    retrieval_similarity
):

    # Account and financial issues can
    # require account-specific investigation.
    if intent in [
        "account_issue",
        "payment_issue",
        "refund_billing"
    ]:
        return "ESCALATE"

    # Low classifier confidence
    if classifier_confidence < 0.60:
        return "ESCALATE"

    # Weak historical evidence
    if retrieval_similarity < 0.25:
        return "ESCALATE"

    return "AUTO"


# ============================================================
# 8. RESPONSE GENERATION
# ============================================================

def generate_response(
    message,
    intent,
    retrieval_results
):

    response, similarity = select_best_response(
        retrieval_results
    )

    # --------------------------------------------------------
    # No reliable historical response
    # --------------------------------------------------------

    if response is None:

        return (
            "I'm sorry you're experiencing this. "
            "Please contact our support team so "
            "we can investigate the issue further."
        ), similarity

    # --------------------------------------------------------
    # Weak retrieval
    # --------------------------------------------------------

    if similarity < 0.25:

        return (
            "I'm sorry you're experiencing this. "
            "We need to look into the issue further. "
            "Please contact our support team for assistance."
        ), similarity

    # --------------------------------------------------------
    # Add a small context-specific introduction
    # --------------------------------------------------------

    introductions = {

        "delivery_issue":
            "I understand you're having a delivery issue. ",

        "order_issue":
            "I understand you're having an issue with your order. ",

        "refund_billing":
            "I understand you're having a refund or billing issue. ",

        "return_issue":
            "I understand you're having an issue with a return. ",

        "account_issue":
            "I understand you're having trouble with your account. ",

        "payment_issue":
            "I understand you're having a payment-related issue. ",

        "device_technical":
            "I understand you're having a technical issue with your device. ",

        "prime_membership":
            "I understand you're having an issue with your Prime membership. ",

        "general_complaint":
            "I understand your concern. "
    }

    introduction = introductions.get(
        intent,
        "I understand your concern. "
    )

    final_response = (
        introduction +
        response
    )

    return (
        final_response,
        similarity
    )


# ============================================================
# 9. COMPLETE SUPPORT AGENT
# ============================================================

def run_agent(message):

    # Step 1: Intent
    intent, confidence = predict_intent(
        message
    )

    # Step 2: Historical retrieval
    retrieved = retrieve_similar_cases(
        message,
        top_k=5
    )

    # Step 3: Select useful response
    _, retrieval_similarity = select_best_response(
        retrieved
    )

    # Step 4: Decide AUTO / ESCALATE
    decision = decide_handling(
        intent,
        confidence,
        retrieval_similarity
    )

    # Step 5: Generate response
    response, _ = generate_response(
        message,
        intent,
        retrieved
    )

    return {
        "intent": intent,
        "confidence": confidence,
        "decision": decision,
        "response": response,
        "retrieval_similarity": retrieval_similarity,
        "evidence": retrieved
    }


# ============================================================
# 10. INTERACTIVE AGENT
# ============================================================

print("\n" + "=" * 70)
print("HIVER AI SUPPORT AGENT")
print("=" * 70)

print("\nEnter a customer message.")
print("Type 'exit' to stop.")


while True:

    message = input(
        "\nCustomer: "
    ).strip()

    if message.lower() == "exit":
        break

    if not message:
        continue

    result = run_agent(
        message
    )

    print("\n" + "=" * 70)
    print("AGENT RESULT")
    print("=" * 70)

    print(
        f"\nIntent: "
        f"{result['intent']}"
    )

    print(
        f"Classifier confidence: "
        f"{result['confidence']:.3f}"
    )

    print(
        f"Retrieval similarity: "
        f"{result['retrieval_similarity']:.3f}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )

    print(
        "\nResponse:"
    )

    print(
        result["response"]
    )

    print(
        "\nTop historical evidence:"
    )

    for i, evidence in enumerate(
        result["evidence"][:3],
        start=1
    ):

        print(
            f"\n{i}. Similarity: "
            f"{evidence['similarity']:.3f}"
        )

        print(
            "Customer:"
        )

        print(
            evidence["customer"][:250]
        )

        print(
            "AmazonHelp:"
        )

        print(
            clean_response(
                evidence["response"]
            )[:350]
        )

        print(
            "\n" + "-" * 70
        )