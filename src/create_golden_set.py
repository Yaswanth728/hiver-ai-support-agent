import pandas as pd

# Load AmazonHelp conversations
df = pd.read_csv("data/amazonhelp_conversations.csv")

# Select 200 random conversations
sample = df.sample(
    n=200,
    random_state=42
).copy()

# Keep only the information needed for manual labeling
golden_set = sample[
    ["tweet_id", "text", "response_text"]
].copy()

# Add an empty intent column
golden_set["intent"] = ""

# Add an empty escalation label
golden_set["expected_decision"] = ""

# Add an empty reason
golden_set["decision_reason"] = ""

# Save golden evaluation set
golden_set.to_csv(
    "evaluation/golden_set.csv",
    index=False
)

print("Golden evaluation set created!")
print("Number of examples:", len(golden_set))
print("File: evaluation/golden_set.csv")