# Causal Inference

## Objective

Estimate causal treatment effects beyond naive correlation, and validate estimates against known ground-truth effects using the IHDP benchmark.

## Dataset

IHDP (Infant Health and Development Program) - ihdp_data.csv

A semi-synthetic benchmark dataset widely used in causal inference research:
- Real covariates from a randomized trial (25 covariates: x1 through x25)
- Simulated potential outcomes (mu0, mu1) with known ground truth
- Binary treatment indicator
- Factual observed outcome (y_factual)

Because the potential outcomes are simulated, the true Individual Treatment Effect (ITE) for every individual is available: 	rue_ite = mu1 - mu0. This enables direct validation of causal estimates — something real observational data cannot provide.

## Methods

### Step 1: Naive Estimate (Biased Baseline)

Computed the naive Average Treatment Effect (ATE) by directly subtracting the mean outcome of control units from the mean outcome of treated units:


aive_ATE = mean(Y | T=1) - mean(Y | T=0)

Compared against the true ATE (mean(true_ite)) to measure confounding bias.

### Step 2: Propensity Score Estimation

Fitted a logistic regression model on the 25 covariates to predict treatment assignment, producing a propensity score for each individual: the probability of receiving treatment given their covariates.

Visualized propensity score distributions for the treated and control groups to inspect the overlap (positivity) assumption.

### Step 3: Propensity Score Matching

Used 1-nearest-neighbor matching on the propensity score: for each treated individual, found the closest control individual. Estimated ATE as the mean difference in factual outcomes between matched treated and matched control units.

Compared matched ATE against naive ATE and true ATE to show that correcting for confounding reduces bias.

### Step 4: Causal Forest (Individual Treatment Effects)

Used EconML's CausalForestDML with a RandomForestRegressor for the outcome model and LogisticRegression for the treatment model. Fitted on the full dataset and estimated individual-level treatment effects (ITEs).

Computed the Causal Forest ATE (mean of estimated ITEs) and compared it to the true ATE.

### Step 5: ITE Validation

- Plotted estimated ITEs against true ITEs with an ideal diagonal reference line.
- Computed ITE RMSE and Pearson correlation between estimated and true ITEs.

### Step 6: ATE Comparison Table

Summarized all three methods:

| Method             | ATE   | Bias       |
|--------------------|-------|------------|
| Naive              | ...   | ...        |
| Propensity Matching| ...   | ...        |
| Causal Forest      | ...   | ...        |

## Key Observations

- The naive ATE is noticeably biased due to confounding: treated and untreated groups differ systematically in their covariates.
- Propensity score matching reduces the bias by comparing similar individuals, but 1-NN matching is imperfect and some residual bias remains.
- The Causal Forest provides the most accurate ATE estimate and additionally produces individual-level treatment effect estimates.
- The scatter plot of estimated vs. true ITEs shows how well the causal forest captures treatment effect heterogeneity across individuals.
- This task illustrates the central challenge of causal inference: estimating what would have happened under a counterfactual (the treatment not taken), which is never directly observed.
