import pandas as pd
import sys
import os

# Handle emojis and non-English text in Windows terminal
sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------
# Files
# --------------------------------------------------

input_file = "evaluation/golden_set.csv"
output_file = "evaluation/golden_set_labeled.csv"

# --------------------------------------------------
# Load original golden set
# --------------------------------------------------

if os.path.exists(output_file):
    df = pd.read_csv(output_file, dtype=str)
    print("Continuing from previously saved labels...")
else:
    df = pd.read_csv(input_file, dtype=str)
    print("Starting golden-set labeling...")

# Make sure label columns are text
df["intent"] = df["intent"].fillna("").astype(str)
df["expected_decision"] = df["expected_decision"].fillna("").astype(str)
df["decision_reason"] = df["decision_reason"].fillna("").astype(str)

# --------------------------------------------------
# Intent options
# --------------------------------------------------

intents = {
    "1": "delivery_issue",
    "2": "order_issue",
    "3": "refund_billing",
    "4": "return_issue",
    "5": "account_issue",
    "6": "payment_issue",
    "7": "device_technical",
    "8": "prime_membership",
    "9": "general_complaint"
}

# --------------------------------------------------
# Label examples
# --------------------------------------------------

for index in range(len(df)):

    # Skip already-labelled examples
    if df.loc[index, "intent"].strip() != "":
        continue

    print("\n" + "=" * 70)
    print(f"EXAMPLE {index + 1} / {len(df)}")
    print("=" * 70)

    print("\nCUSTOMER:")
    print(df.loc[index, "text"])

    print("\nHISTORICAL AMAZONHELP RESPONSE:")
    print(df.loc[index, "response_text"])

    # --------------------------------------------------
    # Intent
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("CHOOSE INTENT")
    print("-" * 70)

    for number, intent in intents.items():
        print(f"{number}. {intent}")

    while True:
        choice = input("\nYour choice (1-9): ").strip()

        if choice in intents:
            break

        print("Please enter a number from 1 to 9.")

    df.loc[index, "intent"] = intents[choice]

    # --------------------------------------------------
    # Auto or escalation
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("CHOOSE HANDLING")
    print("-" * 70)

    print("A. AUTO")
    print("B. ESCALATE")

    while True:
        decision = input("\nYour choice (A/B): ").strip().upper()

        if decision in ["A", "B"]:
            break

        print("Please enter A or B.")

    if decision == "A":

        df.loc[index, "expected_decision"] = "AUTO"
        df.loc[index, "decision_reason"] = (
            "Low-risk and suitable for automated handling."
        )

    else:

        df.loc[index, "expected_decision"] = "ESCALATE"

        print("\nEscalation reason:")
        print("1. Requires account-specific investigation")
        print("2. Financial/refund issue")
        print("3. Security/privacy concern")
        print("4. Complaint or complex unresolved issue")

        reasons = {
            "1": "Requires account-specific investigation.",
            "2": "Financial or refund issue requires human review.",
            "3": "Security or privacy concern requires human review.",
            "4": "Complaint or complex unresolved issue requires human review."
        }

        while True:
            reason = input("\nReason (1-4): ").strip()

            if reason in reasons:
                break

            print("Please enter a number from 1 to 4.")

        df.loc[index, "decision_reason"] = reasons[reason]

    # --------------------------------------------------
    # Save after every example
    # --------------------------------------------------

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved successfully!")
    print(f"Progress: {index + 1}/{len(df)}")

print("\n" + "=" * 70)
print("LABELING COMPLETE!")
print("=" * 70)
print(f"Labeled file: {output_file}")