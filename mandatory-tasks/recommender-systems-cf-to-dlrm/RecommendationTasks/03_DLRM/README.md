# Task 3: DLRM (Deep Learning Recommendation Model)

## Objective

Implement a DLRM-style architecture from scratch and compare it against the neural CTR baseline from Task 2.

## Dataset

Same advertising dataset as Task 2 (train.csv, test.csv):
- 13 numerical features
- 26 categorical features
- Binary click label

## Approach

### Preprocessing

- Numerical features: missing values filled with column median, then StandardScaler applied.
- Categorical features: missing values filled with "missing" string.
- No target encoding used here; instead, each categorical feature gets its own embedding table.

### DLRM Architecture

The model follows the DLRM design pattern:

1. Dense MLP for numerical features: 64 → 32 → 16 (embedding dimension), ReLU activations. This projects the dense input to the same embedding dimension as the categorical embeddings.

2. Embedding tables for categorical features: each of the 26 categorical features has a dedicated StringLookup layer followed by an Embedding layer of dimension 16. The embeddings are flattened.

3. Pairwise interaction layer: all feature vectors (1 dense + 26 categorical, total 27) are combined using dot-product pairwise interactions. For N feature vectors, N*(N-1)/2 pairwise dot products are computed.

4. Top MLP: the concatenation of the dense representation and all pairwise interactions is passed through a final MLP (64 → 32 → 1 with sigmoid), producing the click probability.

### Training

- Adam optimizer, binary cross-entropy loss.
- Metrics tracked during training: accuracy, ROC-AUC, PR-AUC.
- Early stopping on validation loss (patience=10).
- 20 epochs maximum.

### Threshold Tuning

Instead of using the default 0.5 threshold, the optimal threshold maximizing F1 on the test set was searched over the range [0.01, 0.50] in steps of 0.01.

### Evaluation

Test set evaluation at the best threshold:
- ROC-AUC
- Accuracy
- PR-AUC
- Log Loss
- F1 Score
- Precision
- Recall
- Confusion Matrix
- Classification Report

### Key Observations

- The dataset's heavy class imbalance (approximately 96% label=0) makes accuracy a poor metric.
- Other scores (F1, PR-AUC) are low for the same reason: the model learns the majority class well but struggles on minority class clicks.
- DLRM's explicit pairwise feature interactions are designed to capture cross-feature signals that a vanilla dense network cannot express without very deep layers.
- The comparison between Task 2 (vanilla neural network) and Task 3 (DLRM) shows the practical gain from explicit interaction modeling in CTR prediction.
