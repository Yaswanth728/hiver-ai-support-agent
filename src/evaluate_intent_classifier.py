import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = joblib.load(
    "experiments/intent_classifier.joblib"
)

print("Intent classifier loaded successfully!")


# ============================================================
# 2. LOAD GOLDEN EVALUATION SET
# ============================================================

golden = pd.read_csv(
    "evaluation/golden_set_labeled.csv",
    dtype=str
)

golden["text"] = golden["text"].fillna("")
golden["intent"] = golden["intent"].fillna("")

print(
    "Golden examples:",
    len(golden)
)


# ============================================================
# 3. PREDICT
# ============================================================

predictions = model.predict(
    golden["text"]
)


# ============================================================
# 4. ACCURACY
# ============================================================

accuracy = accuracy_score(
    golden["intent"],
    predictions
)

print("\n" + "=" * 60)
print("GOLDEN SET RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)


# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        golden["intent"],
        predictions,
        zero_division=0
    )
)


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

labels = sorted(
    golden["intent"].unique()
)

cm = confusion_matrix(
    golden["intent"],
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)


# ============================================================
# 7. SAVE PREDICTIONS
# ============================================================

golden["predicted_intent"] = predictions

golden["correct"] = (
    golden["intent"] ==
    golden["predicted_intent"]
)

golden.to_csv(
    "evaluation/intent_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nPredictions saved to:"
)

print(
    "evaluation/intent_predictions.csv"
)