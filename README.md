# AI Support Agent for AmazonHelp

## 1. Project Overview

This project implements an AI-powered customer support agent for AmazonHelp using the Customer Support on Twitter dataset from Kaggle.

The agent performs three main tasks:

1. Classifies an incoming customer message into a support intent.
2. Retrieves a historically similar AmazonHelp conversation to ground the response.
3. Decides whether the issue can be handled automatically or should be escalated.

The goal is not to generate arbitrary responses, but to ground support replies in how AmazonHelp historically responded to similar customer issues.

---

## 2. Dataset

The project uses the Customer Support on Twitter dataset.

Dataset columns include:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

The dataset contains approximately 2.8 million tweets.

AmazonHelp was selected because it was one of the largest support accounts in the dataset, with approximately 170,000 tweets.

From these tweets, customer → AmazonHelp response pairs were extracted.

### Data processing

Original AmazonHelp conversations:

**168,814**

After removing missing values, duplicate customer messages and very short messages:

**152,983**

These conversations were used as the historical support corpus.

---

## 3. Intent Taxonomy

The project uses 11 intents:

| Intent | Description |
|---|---|
| `delivery_issue` | Late, missing or incorrectly delivered packages |
| `order_issue` | Order status, cancellation and order problems |
| `return_refund` | Returns, refunds, replacements and damaged items |
| `payment_billing` | Charges, payments and billing issues |
| `product_device_issue` | Kindle, Echo, Alexa, Fire TV and device problems |
| `digital_service_issue` | Prime Video and other digital-service problems |
| `account_security` | Account access, suspicious messages and security issues |
| `seller_issue` | Third-party seller and marketplace problems |
| `information_feedback` | Product information, availability, promotions and feedback |
| `general_support` | Support requests that do not fit another category |
| `non_support` | Thanks, greetings, compliments and casual messages |

---

## 4. System Architecture

The agent consists of three major components.

### Intent Classifier

A TF-IDF based Logistic Regression classifier is used.

Features include:

- Word n-grams (1–2)
- Character n-grams
- Sublinear TF-IDF weighting
- Class-balanced Logistic Regression

### Historical Reply Retrieval

For each customer message, TF-IDF similarity is used to retrieve historically similar customer messages and their AmazonHelp replies.

The retrieved conversation provides evidence for generating a grounded response.

### Escalation

A rule-based escalation component identifies cases that should receive additional human attention.

Examples include:

- Security-related problems
- Seller disputes
- Repeated unresolved problems
- Delivery failures
- Refund disputes
- Strong complaints
- Explicit requests for human assistance

---

## 5. Evaluation Methodology

A manually reviewed golden set of 200 customer messages was used as the held-out evaluation set.

The classifier was trained on the weakly labelled historical dataset while golden-set examples were excluded from the training corpus.

Retrieval evaluation also removed:

- Golden-set tweet IDs
- Exact duplicate customer messages

This prevents direct test-set leakage.

---

## 6. Results

### Intent Classification

| Method | Accuracy |
|---|---:|
| Majority-class baseline | 30.50% |
| Initial TF-IDF classifier | 30.00% |
| Improved classifier, 5-fold CV | 36.00% |
| Final held-out classifier | **52.50%** |

The final classifier correctly classified:

**105 / 200 messages**

and incorrectly classified:

**95 / 200 messages**.

The final model therefore improves substantially over the majority baseline.

---

### Escalation

The rule-based escalation system achieved:

**59.50% accuracy**

on the 200-example golden set.

It correctly identified 54 of the 124 examples labelled for escalation.

The main challenge was identifying unresolved issues that were expressed indirectly.

---

### Retrieval

Leakage-free retrieval results:

| Metric | Result |
|---|---:|
| Average similarity | **0.5991** |
| Median similarity | **0.4916** |
| Minimum | 0.0000 |
| Maximum | 1.0000 |

The results show that many messages have useful lexical matches, but high similarity does not always mean that the retrieved response is actually useful.

---

## 7. LLM-as-Judge Evaluation

A sample of 50 retrieval results was evaluated using a rubric covering:

- Relevance
- Helpfulness
- Groundedness
- Tone

Each category was scored from 0–2.

Results:

| Metric | Score |
|---|---:|
| Average relevance | 1.20 / 2 |
| Average helpfulness | 1.16 / 2 |
| Average groundedness | 2.00 / 2 |
| Average tone | 2.00 / 2 |
| Average total | 6.36 / 8 |
| Pass rate | **76%** |

The evaluation was performed by a single LLM-based judge. Therefore, human inter-rater agreement was not measured and the results should not be described as human agreement.

---

## 8. Top Failure Modes

### 1. Delivery vs. General Support Confusion

Many delivery-related complaints contain broad phrases such as "please help" or "customer service", causing the classifier to predict `general_support`.

**Hypothesis:** More explicit examples separating delivery problems from generic support requests are needed.

### 2. Information vs. Support Requests

Questions about promotions, availability and shipping information can resemble active support problems.

**Hypothesis:** Better intent definitions and more representative training examples would reduce this confusion.

### 3. Rare Intents

`account_security` and `seller_issue` have relatively few examples.

The classifier frequently predicts more common classes instead.

**Hypothesis:** Class-balanced training alone is insufficient when the underlying labelled examples are sparse.

### 4. Multilingual and Encoding Problems

Several non-English messages became incorrectly encoded during CSV processing.

This resulted in unreadable text and very low retrieval similarity for some examples.

**Hypothesis:** Consistent UTF-8 handling and multilingual embeddings would substantially improve these cases.

### 5. Lexical Similarity Does Not Guarantee Useful Retrieval

Some retrieved messages had high TF-IDF similarity but addressed a different problem.

For example, two messages may share words such as "delivery", "package" or "Amazon" while requiring different resolutions.

**Hypothesis:** Semantic embeddings and cross-encoder reranking would provide better retrieval quality.

---

## 9. What Is Misleading About My Headline Number?

The headline classification accuracy of **52.50%** should not be interpreted as production-level agent accuracy.

The test set contains only 200 examples and some intents are rare. In addition, the classifier was trained using weakly labelled historical data, while the evaluation labels were manually reviewed.

The dataset also contains multilingual and encoding issues.

Therefore, 52.50% is a useful held-out benchmark for this prototype, but it does not mean that the complete support agent would correctly solve 52.5% of real customer conversations.

---

## 10. One-Week Improvement Plan

### Days 1–2
Improve data quality and ensure all text is consistently stored as UTF-8.

### Days 2–3
Create additional labelled examples for rare intents such as seller and account-security issues.

### Days 3–4
Replace TF-IDF retrieval with multilingual sentence embeddings.

### Day 5
Add a reranking stage to select the most useful historical response.

### Day 6
Improve escalation using confidence scores and intent-specific rules.

### Day 7
Expand the golden set and repeat the complete evaluation.

---

## 11. Limitations

The current prototype has several limitations:

- Only one brand was evaluated.
- The golden set contains 200 examples.
- Weak supervision introduces label noise.
- TF-IDF does not capture semantic similarity well.
- Multilingual messages are affected by encoding issues.
- Human inter-rater agreement was not available.
- The escalation system is rule-based.

These limitations are important when interpreting the reported metrics.

---

## 12. Reproducibility

The project is organized into separate scripts for:

- Data extraction
- Data cleaning
- Golden-set validation
- Weak labelling
- Intent classification
- Retrieval
- Escalation
- Evaluation

The main datasets and intermediate results are stored under `data/` and `results/`.

The complete pipeline can be reproduced by installing the dependencies listed in `requirements.txt` and running the scripts in the documented order.

---

## 13. Conclusion

The prototype demonstrates an end-to-end AI support workflow combining intent classification, historical-response retrieval and escalation.

The final classifier achieved **52.50% accuracy**, improving over the **30.50% majority baseline**. Retrieval achieved an average leakage-free similarity of **0.5991**, while the escalation component achieved **59.50% accuracy**.

The LLM-based evaluation found that **76% of sampled retrieved responses passed the quality threshold**.

The results demonstrate that historical customer-support conversations can provide useful grounding for an AI support agent, while also highlighting the need for better semantic retrieval, multilingual handling, more labelled data and stronger escalation logic.