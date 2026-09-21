# Hiver AI Support Agent

An AI-powered customer support agent built using historical AmazonHelp customer-service conversations.

The system combines:

- Intent classification
- Historical conversation retrieval
- Response generation using historical support evidence
- Automatic escalation decisions
- Evaluation of classification and retrieval performance

## Project Overview

The goal of this project is to build an AI support agent that can process customer-service queries and determine how they should be handled.

Given a customer message, the system:

1. Classifies the customer's intent.
2. Retrieves similar historical conversations.
3. Uses historical AmazonHelp responses as supporting evidence.
4. Generates a support response.
5. Determines whether the request can be handled automatically or should be escalated.

## Dataset

The primary dataset contains:

- **136,221 historical conversations**
- Customer messages
- AmazonHelp responses
- Tweet/conversation identifiers
- Conversation metadata

The project also uses the Twitter Customer Support (TWCS) dataset for exploration and experimentation.

## Intent Taxonomy

The system supports nine customer-support intents:

| Intent | Description |
|---|---|
| `delivery_issue` | Delivery delays, missing or incorrect deliveries |
| `order_issue` | Problems related to orders and order status |
| `refund_billing` | Refund requests and billing-related concerns |
| `return_issue` | Product returns and return processing |
| `account_issue` | Login, account access, or account-related problems |
| `payment_issue` | Payment and transaction problems |
| `device_technical` | Technical issues with Amazon devices |
| `prime_membership` | Amazon Prime membership-related issues |
| `general_complaint` | General complaints and unsupported issues |

Detailed definitions are available in:

`report/intent_taxonomy.md`

## System Architecture

```text
                         Customer Message
                                |
                                v
                     +----------------------+
                     |   Intent Classifier  |
                     +----------+-----------+
                                |
                                v
                         Predicted Intent
                                |
                                v
                     +----------------------+
                     | Historical Retrieval |
                     |    136K+ Records     |
                     +----------+-----------+
                                |
                                v
                    Similar Conversations
                                |
                                v
                    Historical Responses
                                |
                                v
                     +----------------------+
                     |   Response Layer     |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |   Decision Layer     |
                     +----------+-----------+
                                |
                    +-----------+-----------+
                    |                       |
                  AUTO                   ESCALATE
