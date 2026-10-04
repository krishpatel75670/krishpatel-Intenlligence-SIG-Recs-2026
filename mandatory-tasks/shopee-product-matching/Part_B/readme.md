# Part B: Text-Based Product Matching

## Objective

Build a product matching system using product title text and compare different text representations.

## Pipeline

Product Titles -> Preprocessing -> Text Representation -> Cosine Similarity -> Threshold -> Match / No Match

## Preprocessing

- Removed non-alphabetic characters (kept only letters and spaces).
- Lowercased all titles.
- Removed English stopwords using NLTK.
- Applied WordNet lemmatization (verb form).

## Pair Construction

- 30,000 positive pairs sampled from within the same label group.
- 30,000 negative pairs sampled from different label groups.
- A separate 10,000-pair evaluation set was held out (not used for threshold tuning).
- Final evaluation used an additional independent 10,000-pair set (5,000 positive, 5,000 negative).

## Experiment 1: TF-IDF (Baseline)

- TF-IDF vectorizer with unigrams and bigrams, max 2,500 features.
- Cosine similarity between pairs.
- Threshold of 0.5 used for the binary match decision.
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.

## Experiment 2: Word2Vec Embeddings

- Trained a Word2Vec skip-gram model (vector size=100, window=5, min_count=2, 20 epochs) on the preprocessed training titles.
- Sentence embedding = mean of token embeddings (out-of-vocabulary tokens contribute zero vectors).
- Threshold sweep from 0.5 to 1.0 (step 0.02) on the tuning set to find the best F1 threshold.
- Evaluated at the best threshold on the independent evaluation set.
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.

## Comparison

Both methods were evaluated on the same evaluation pairs:
- TF-IDF at threshold 0.5
- Word2Vec at its best threshold from validation

Results showed Precision and Recall tradeoffs between the two methods.

## Key Observations

- TF-IDF is strong for keyword-heavy titles where exact vocabulary overlap is a useful signal.
- Word2Vec captures some semantic similarity but the short, noisy nature of product titles limits its advantage.
- Threshold selection significantly affects precision and recall: a lower threshold increases recall (finds more matches) at the cost of more false positives.
- Text-only matching is limited by the challenge identified in Part A: the same product can have very different wording, and different products can have very similar wording.
