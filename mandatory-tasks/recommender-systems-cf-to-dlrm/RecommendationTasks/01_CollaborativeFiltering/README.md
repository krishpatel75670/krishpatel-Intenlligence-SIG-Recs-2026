# Task 1: Collaborative Filtering

## Objective

Build and compare two collaborative filtering approaches — memory-based and model-based — using the same interaction data and evaluation protocol.

## Dataset

The MovieLens 100K dataset was used (moviedata.csv), which contains user-item ratings on a 1-5 star scale. The dataset has 943 users and 1,682 movies.

## Approach

### Data Exploration and Preprocessing

- Loaded and validated the dataset: checked for missing values, duplicate user-item pairs, rating distribution, and ratings per user.
- Retained only user, item, and ating columns. Converted all to appropriate types.
- Computed and visualized the interaction matrix sparsity.

### Train / Validation / Test Split

A per-user split was applied (80% train, 10% validation, 10% test) to ensure every user has at least one training rating. No test-set tuning was performed.

### Part A: Memory-Based Collaborative Filtering (User-User)

- Built a user-user cosine similarity matrix from the mean-centered training interaction matrix.
- For each target user and unrated item, predicted the rating using a similarity-weighted average of neighbors' deviation from their mean.
- Used validation RMSE to select the best number of neighbors k from the set {5, 10, 20, 30, 50}.
- Evaluated the best k on the untouched test set.

### Part B: Model-Based Collaborative Filtering (Matrix Factorization)

- Implemented matrix factorization from scratch using stochastic gradient descent (SGD) with L2 regularization.
- The model learns user biases, item biases, and latent factor vectors.
- Searched over a small set of factor sizes (20, 30, 50) using validation RMSE.
- Used early stopping on validation RMSE with patience=3 epochs.
- Trained the best configuration and evaluated on the untouched test set.

### Evaluation Metrics

RMSE and MAE were reported for:
- Global Mean Baseline
- Memory-Based User CF (best k)
- Matrix Factorization (best factor size)

### Example Recommendations

The final MF model was used to generate top-10 recommendations for an example user by predicting ratings for all unrated items and ranking them.

## Key Observations

- Memory-based CF is intuitive but depends heavily on neighborhood overlap; sparse data hurts its quality.
- Matrix factorization can discover hidden preference patterns and is more compact, but requires hyperparameter choices and is less interpretable.
- MF generally achieved lower RMSE on this dataset due to its ability to handle sparsity through latent representations.
