# Hiver AI Support Agent
## Final Project Report

---

## 1. Introduction

Customer-support teams handle a large number of repetitive customer queries involving deliveries, orders, payments, refunds, returns, accounts, devices, and subscriptions.

This project explores an AI-assisted customer-support agent that uses historical customer-service conversations to understand customer issues and provide grounded responses.

The system combines intent classification, historical conversation retrieval, response generation, and escalation logic.

---

## 2. Objective

The main objectives are:

- Analyze historical customer-support conversations.
- Develop a support-intent taxonomy.
- Classify incoming customer messages.
- Retrieve similar historical conversations.
- Use historical support responses as evidence.
- Generate grounded responses.
- Automatically escalate sensitive or uncertain cases.
- Evaluate the system using a manually labelled golden dataset.

---

## 3. Dataset

The primary AmazonHelp dataset contains:

```text
136,221 conversations