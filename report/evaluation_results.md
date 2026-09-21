# Evaluation Results

## 1. Evaluation Dataset

A manually labelled golden evaluation set containing 200 examples was created.

The examples cover the project's nine support intents.

## 2. Intent Classification

The initial intent-classification experiment produced the following result:

| Metric | Result |
|---|---:|
| Evaluation examples | 200 |
| Accuracy | 6% |
| Macro-level performance | Low |

The result indicates that the initial classifier is not sufficiently reliable for production use.

The classifier is therefore treated as a baseline experiment rather than a final production model.

## 3. Retrieval Evaluation

The retrieval system searches the historical AmazonHelp conversation corpus using TF-IDF and cosine similarity.

The historical corpus contains:

```text
136,221 conversations