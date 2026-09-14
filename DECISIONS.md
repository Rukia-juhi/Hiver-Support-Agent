# Project Decision Log

## Decision 1 — Select AmazonHelp

AmazonHelp was selected because it has a large number of conversations in the dataset, providing enough historical examples for training and retrieval.

## Decision 2 — Use customer → support conversation pairs

Only conversations where an inbound customer message was followed by an AmazonHelp response were used for the historical support corpus.

This provides both the original customer problem and the actual historical support response.

## Decision 3 — Use 11 intents

The original dataset does not provide a clean intent taxonomy. Eleven practical support categories were created to balance coverage and classification difficulty.

## Decision 4 — Create a manually reviewed golden set

A 200-example golden set was created to provide a fixed evaluation benchmark.

The labels were reviewed and corrected before final evaluation.

## Decision 5 — Use weak supervision

Because manually labelling more than 150,000 conversations was impractical, keyword-based weak labelling was used to create a larger training set.

This introduces noise but allows the classifier to learn from substantially more data.

## Decision 6 — Use TF-IDF

TF-IDF was selected as a simple, interpretable baseline that is computationally inexpensive and suitable for a prototype.

## Decision 7 — Add character n-grams

Character features were added because customer-support messages contain spelling variations, abbreviations and noisy text.

## Decision 8 — Use class-balanced Logistic Regression

Class balancing was enabled because the intent distribution is highly uneven.

## Decision 9 — Use historical-response retrieval

Rather than generating unsupported answers, the system retrieves similar historical customer conversations and uses their responses as grounding evidence.

## Decision 10 — Evaluate retrieval separately

Retrieval quality was evaluated independently from classification because a correct intent prediction does not guarantee a useful historical response.

## Decision 11 — Remove test leakage

Golden-set tweet IDs and exact duplicate customer messages were removed from the retrieval corpus before final evaluation.

This prevents the evaluation from rewarding exact memorization.

## Decision 12 — Use rule-based escalation

A rule-based escalation layer was chosen because safety-critical cases such as security problems, seller disputes and repeated unresolved issues should have explicit handling rules.

## Decision 13 — Use a 50-example judge sample

A stratified sample of 50 retrieval results was selected for detailed quality evaluation.

The sample includes both high- and low-similarity retrievals.

## Decision 14 — Use an LLM judge rubric

The judge evaluates relevance, helpfulness, groundedness and tone on a 0–2 scale.

A total score of at least 6/8 is considered a pass.

## Decision 15 — Report limitations honestly

The 52.50% classification accuracy is reported as a held-out benchmark, not as production performance.

The LLM judge evaluation is also explicitly described as a single-rater evaluation because an independent human rater was unavailable.