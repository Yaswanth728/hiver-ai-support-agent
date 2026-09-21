import pandas as pd

# Load our AmazonHelp conversation dataset
df = pd.read_csv("data/amazonhelp_conversations.csv")

print("AmazonHelp conversation dataset")
print("Number of conversations:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nSample customer messages:\n")

for i, message in enumerate(df["text"].head(30), start=1):
    print(f"{i}. {message}")