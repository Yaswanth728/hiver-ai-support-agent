# Hiver AI Support Agent

An AI-powered customer support agent built using historical AmazonHelp customer-service conversations.

The system combines intent classification, historical response retrieval, grounded response generation, and escalation logic to assist with customer-support queries.

## Project Overview

The objective of this project is to build a support agent that can:

- Understand customer support messages
- Classify customer intent
- Retrieve similar historical conversations
- Use historical AmazonHelp responses as evidence
- Generate a grounded support response
- Decide whether a case can be handled automatically or should be escalated

## Dataset

The primary dataset contains:

- 136,221 historical conversations
- Customer messages
- AmazonHelp responses
- Tweet/conversation identifiers
- Conversation metadata

The project also uses the Twitter Customer Support (TWCS) dataset for exploration and experimentation.

## Intent Taxonomy

The system uses the following support intents:

1. `delivery_issue`
2. `order_issue`
3. `refund_billing`
4. `return_issue`
5. `account_issue`
6. `payment_issue`
7. `device_technical`
8. `prime_membership`
9. `general_complaint`

Detailed definitions are available in:

`report/intent_taxonomy.md`

## System Architecture

```text
                    Customer Message
                           |
                           v
                  +------------------+
                  | Intent Classifier|
                  +--------+---------+
                           |
                           v
                     Intent Label
                           |
                           v
                  +------------------+
                  | Historical       |
                  | Retrieval        |
                  | 136K+ Messages   |
                  +--------+---------+
                           |
                           v
                  Similar Conversations
                           |
                           v
                  Historical Responses
                           |
                           v
                  +------------------+
                  | Response Layer   |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Decision Layer   |
                  +--------+---------+
                           |
                  +--------+--------+
                  |                 |
                AUTO            ESCALATE

##Project structure