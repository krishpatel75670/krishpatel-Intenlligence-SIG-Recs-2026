# Task 2: Neural CTR Prediction

## Objective

Train a neural network for binary click-through rate (CTR) prediction on a tabular advertising dataset.

## Dataset

The supplied advertising dataset was used (train.csv and test.csv), which contains:
- 13 integer numerical features
- 26 categorical features
- Binary label (1 = clicked, 0 = not clicked)

The dataset has significant class imbalance: approximately 96% of labels are 0.

## Approach

### Preprocessing

- Numerical features: missing values filled with the column median.
- Categorical features: missing values filled with the most frequent value (mode).
- High-cardinality categorical features (more than 50 unique values): encoded with TargetEncoder.
- Low-cardinality categorical features: encoded with OneHotEncoder.
- Numerical features: standardized with StandardScaler.
- All transformations were fit on the training split only.

### Architecture

A feedforward neural network with:
- Input layer matching the preprocessed feature count
- Dense layers: 64 → 32 → 16 → 1
- ReLU activations in hidden layers, Sigmoid in output
- Binary cross-entropy loss, Adam optimizer
- Early stopping on validation loss (patience=10) to prevent overfitting

### Class Imbalance Handling

The model was trained twice:
1. Without class weights (default)
2. With balanced class weights computed from training labels

Class weights assign higher penalty to the minority class (clicked = 1) to counteract the imbalance.

### Evaluation Metrics

Both experiments evaluated on the test set:
- ROC-AUC
- PR-AUC
- Accuracy
- Log Loss
- F1 Score
- Precision
- Recall
- Confusion Matrix
- Full Classification Report

### Key Observations

- Accuracy is misleadingly high due to class imbalance: predicting 0 for all samples yields ~96% accuracy.
- ROC-AUC and PR-AUC are more informative metrics for this imbalanced setting.
- Class-weighted training improved recall for the minority class at the cost of some precision.
