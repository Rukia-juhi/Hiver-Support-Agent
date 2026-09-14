# Hiver Support Agent

## 1. Project Overview

This project builds an AI-powered customer support agent for Amazon using the Customer Support on Twitter dataset. The system is designed to classify incoming customer messages, retrieve similar historical conversations, generate a support response, and decide whether the issue should be automatically handled or escalated to a human agent.

The project focuses on building a practical support automation pipeline while evaluating its performance using a manually reviewed golden dataset.

---

## 2. Dataset

The project uses the **Customer Support on Twitter** dataset from Kaggle. The dataset contains approximately 3 million tweets involving customer support interactions between customers and companies.

The dataset contains the following columns:

* `tweet_id`
* `author_id`
* `inbound`
* `created_at`
* `text`
* `response_tweet_id`
* `in_response_to_tweet_id`

Amazon was selected as the target brand because `AmazonHelp` has a large number of customer support interactions.

The raw dataset is not included in the repository because of its large size.

The original dataset contained **2,811,774 tweets**. After extracting AmazonHelp customer-support conversations, approximately **168,814 customer-reply pairs** were obtained. After cleaning and removing very short messages, **152,983 training examples** were available.

---

## 3. How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rukia-juhi/Hiver-Support-Agent.git
cd Hiver-Support-Agent
```

### Step 2: Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the Dataset

Download the **Customer Support on Twitter** dataset from Kaggle and place `twcs.csv` inside the `data/` folder:

```text
data/twcs.csv
```

The raw dataset is not included in this repository because of its size.

### Step 5: Prepare the Data

Run the following commands to extract AmazonHelp conversations, clean the data, and generate weak labels:

```bash
python src/conversations.py
python src/prepare_training_data.py
python src/weak_label_data.py
```

### Step 6: Run the Evaluations

Run the intent classification, escalation, and retrieval evaluations:

```bash
python src/final_evaluation.py
python src/final_escalation_evaluation.py
python src/evaluate_retrieval.py
```

### Step 7: View the Results

The main evaluation results are saved in the `results/` directory:

```text
results/final_intent_predictions.csv
results/final_escalation_results.csv
results/retrieval_evaluation.csv
```

The manually reviewed golden set is available at:

```text
data/golden_set.csv
```
### Step 8: Run the Interactive Support Agent

The project also includes an interactive end-to-end support agent that combines intent classification, historical reply retrieval, and escalation decisions.

Run:

```bash
python src/agent.py
```

Enter a customer message when prompted. The agent returns:

* Predicted intent
* Most similar historical Amazon customer message
* Similarity score
* Historical Amazon reply used as a reference
* Draft reply
* Auto-handle or escalation decision
* Reason for escalation

Example:

```text
Customer message:
My package is late and I still haven't received it.

Intent:
delivery_issue

Decision:
ESCALATE

Reason:
The issue appears unresolved or repeated.
```

The interactive agent is provided as a demonstration of the complete support-agent pipeline. The evaluation results reported in this README are produced by the separate evaluation scripts described above.


---

## 4. Intent Taxonomy

The system uses 11 support intents:

1. **delivery_issue** – Late, missing, delayed, or incorrectly delivered packages.
2. **order_issue** – Order status, cancellation, preorder, or replacement-order problems.
3. **return_refund** – Returns, refunds, damaged items, wrong items, or refund disputes.
4. **payment_billing** – Payment methods, unexpected charges, authorization, or billing issues.
5. **product_device_issue** – Kindle, Echo, Alexa, Fire TV, and other Amazon device issues.
6. **digital_service_issue** – Prime Video, streaming, Kindle content, applications, and other digital services.
7. **account_security** – Account access, suspicious messages, phishing, or security concerns.
8. **seller_issue** – Third-party seller problems and A-to-z Guarantee issues.
9. **information_feedback** – Product availability, pricing, promotions, packaging feedback, and general information.
10. **general_support** – General requests for customer support that do not fit another category.
11. **non_support** – Thanks, greetings, compliments, jokes, casual conversation, and already-resolved issues.

---

## 5. System Architecture

The system consists of four main components:

### 1. Intent Classification

The incoming customer message is classified into one of the predefined support intents.

A TF-IDF based Logistic Regression model is used as the baseline/final prototype classifier.

### 2. Historical Reply Retrieval

The system searches historical Amazon customer-support conversations using TF-IDF similarity.

The most similar historical customer message and its Amazon response are retrieved to provide grounding for the response.

### 3. Escalation Decision

A rule-based escalation component determines whether the issue should be handled automatically or escalated to a human.

Escalation is considered for cases involving security concerns, seller problems, unresolved issues, repeated complaints, delivery problems, refunds, order problems, and strong customer complaints.

### 4. Evaluation

The system is evaluated using a manually reviewed golden set of 200 customer messages.

The evaluation measures:

* Intent classification accuracy and F1-score
* Retrieval similarity and quality
* Escalation accuracy
* LLM-as-judge response quality

---

## 6. Evaluation Methodology

A **200-example manually reviewed golden set** was created from Amazon customer-support conversations.

The golden set contains:

* Customer message
* Historical Amazon reply
* Intent label
* Escalation decision
* Escalation reason

The classifier was evaluated on these 200 examples while excluding golden-set tweet IDs from the training data.

For retrieval evaluation, both golden-set tweet IDs and exact duplicate customer messages were removed from the historical retrieval corpus. This provides a **leakage-reduced retrieval evaluation**.

The evaluation is intended as a prototype benchmark rather than a production-level estimate.

---

## 7. Results

### Intent Classification

Three approaches were compared:

| Method                                             |   Accuracy |
| -------------------------------------------------- | ---------: |
| Majority-class baseline                            |     30.50% |
| Initial TF-IDF classifier                          |     30.00% |
| Improved classifier with word + character features |     36.00% |
| Final weakly supervised classifier                 | **52.50%** |

The final classifier achieved:

* **Accuracy:** 52.50%
* **Macro F1:** 0.42
* **Weighted F1:** 0.50
* **Correct predictions:** 105
* **Wrong predictions:** 95
* **Test samples:** 200

The final model improved substantially over the majority baseline, although performance remains limited for several minority intents.

### Retrieval

After removing exact golden-set message duplicates, the retrieval evaluation produced:

* **Average similarity:** 0.5991
* **Median similarity:** 0.4916
* **Minimum similarity:** 0.0000
* **Maximum similarity:** 1.0000

The results show that many customer messages have reasonably similar historical examples, but high text similarity does not always guarantee that the retrieved response is appropriate.

### Escalation

The improved rule-based escalation system achieved:

* **Accuracy:** 59.50%
* **No-escalation F1:** 0.62
* **Escalation F1:** 0.57

It correctly identified **54 of 124 escalation cases**.

This shows that simple rules can provide a useful first-stage safety mechanism, but additional context and better decision boundaries are required.

---

## 8. LLM-as-Judge Evaluation

A sample of 50 retrieval results was evaluated using an LLM-based rubric.

Each response was scored from 0–2 on:

* Relevance
* Helpfulness
* Groundedness
* Tone

Results:

* **Average relevance:** 1.20 / 2
* **Average helpfulness:** 1.16 / 2
* **Average groundedness:** 2.00 / 2
* **Average tone:** 2.00 / 2
* **Average total:** 6.36 / 8
* **Pass rate:** 76% (38/50)

The evaluation suggests that the retrieved historical replies were generally grounded and appropriately toned, while relevance and helpfulness were the main weaknesses.

This was a **single-rater LLM evaluation**, so human agreement statistics such as Cohen's kappa were not measured.

---

## 9. Top Failure Modes

### 1. Overprediction of General Support

The weak-labeling rules assign many messages to `general_support`, causing the classifier to overpredict this category.

**Hypothesis:** Better intent definitions and manually labelled training examples would reduce this problem.

### 2. Minority Intent Performance

Some intents, such as `seller_issue` and `account_security`, have very few examples.

**Hypothesis:** Class imbalance makes it difficult for the classifier to learn reliable decision boundaries.

### 3. Similar Text but Wrong Context

Retrieval can return messages with similar words but different underlying problems.

For example, several delivery-related messages can have extremely high similarity even when the actual customer situation differs.

**Hypothesis:** Semantic embeddings and intent-aware retrieval would improve contextual matching.

### 4. Escalation False Negatives

Some serious issues were classified as non-escalation, including unresolved refunds, repeated support requests, and lost packages.

**Hypothesis:** The rule system needs stronger contextual reasoning instead of relying mainly on keywords.

### 5. Non-English Messages

Some multilingual messages were difficult for the TF-IDF classifier and retrieval system.

**Hypothesis:** A multilingual embedding model would provide better representations for non-English customer messages.

---

## 10. What Is Misleading About My Headline Number?

The headline accuracy of **52.50%** can be misleading if interpreted as production-level performance.

First, the evaluation set contains only **200 examples**, so it is relatively small.

Second, the training data uses **weakly generated labels**, rather than fully human-labelled historical data. Therefore, the model can learn errors introduced by the labeling rules.

Third, the intent distribution is highly imbalanced. Some intents contain many more examples than others.

Finally, the evaluation focuses mainly on intent classification. A production support agent would also need to produce correct replies, make safe escalation decisions, and handle multilingual and ambiguous customer requests.

Therefore, **52.50% should be interpreted as a prototype benchmark, not as an estimate of real-world support automation accuracy.**

---

## 11. One-Week Improvement Plan

### Day 1–2: Improve Dataset Labelling

Create a larger manually labelled dataset with clearer intent definitions and balanced representation across all categories.

### Day 3: Improve Classification

Experiment with sentence embeddings or transformer-based classifiers instead of relying only on TF-IDF.

### Day 4: Improve Retrieval

Use semantic embeddings and combine similarity with intent matching.

### Day 5: Improve Escalation

Replace keyword-only rules with a classifier or LLM-based decision system using explicit safety criteria.

### Day 6: Improve Evaluation

Expand the golden set and introduce independent human reviewers to measure inter-rater agreement.

### Day 7: End-to-End Testing

Evaluate the complete system on unseen customer conversations and analyze errors by intent, language, and escalation severity.

---

## 12. Limitations

The main limitations of the current prototype are:

* The golden set contains only 200 examples.
* Historical training labels are weak labels rather than fully human-labelled data.
* The classifier is based on TF-IDF features.
* Retrieval is based on lexical similarity.
* Escalation uses rules rather than a learned decision model.
* Human agreement was not measured.
* Multilingual messages are not handled optimally.
* The system has not been evaluated in a live production environment.

---

## 13. Reproducibility

The project contains the source code, processed datasets, evaluation scripts, and results required to reproduce the experiments.

The raw `twcs.csv` dataset is intentionally excluded from GitHub because of its size.

The repository contains:

```text
src/
├── conversations.py
├── prepare_training_data.py
├── weak_label_data.py
├── intent_classifier.py
├── improved_classifier.py
├── final_evaluation.py
├── final_escalation_evaluation.py
├── evaluate_retrieval.py
└── create_judge_sample.py
```

The main outputs are stored in:

```text
results/
```

---

## 14. Conclusion

This project demonstrates a complete prototype for an AI-powered customer support agent using real historical customer-service conversations.

The final weakly supervised intent classifier achieved **52.50% accuracy**, improving over the **30.50% majority baseline**. Retrieval showed moderate similarity between incoming messages and historical support conversations, while the escalation system achieved **59.50% accuracy**.

The LLM-based evaluation achieved a **76% pass rate**, with strong groundedness and tone but weaker relevance and helpfulness.

The results demonstrate that historical customer-support data can provide useful grounding for an automated support agent, but substantial improvements in data quality, semantic retrieval, intent classification, and escalation reasoning are required before deployment in a production environment.
