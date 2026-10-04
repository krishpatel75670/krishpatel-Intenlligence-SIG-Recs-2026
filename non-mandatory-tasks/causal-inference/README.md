# Causal Inference — IHDP Benchmark

## Objective

Estimate causal treatment effects beyond naive correlation, and validate estimates against known ground-truth effects using the IHDP benchmark.

## File Structure

```
causal-inference/
├── README.md          # This file
├── ihdp_data.csv      # IHDP semi-synthetic benchmark dataset
└── notebook.ipynb     # Full analysis: EDA → methods → evaluation
```

## Dataset

**IHDP (Infant Health and Development Program)** — `ihdp_data.csv`

A semi-synthetic benchmark dataset widely used in causal inference research:

| Column | Description |
|---|---|
| `x1` – `x25` | 25 real covariates from the original randomized trial |
| `treatment` | Binary treatment indicator (1 = treated, 0 = control) |
| `y_factual` | Factual observed outcome |
| `mu0` | Simulated potential outcome under control |
| `mu1` | Simulated potential outcome under treatment |

Because the potential outcomes (`mu0`, `mu1`) are simulated, the **true Individual Treatment Effect (ITE)** for every individual is available:

```
true_ite = mu1 - mu0
```

This enables direct validation of causal estimates — something real observational data cannot provide.

## Requirements

```
pandas
numpy
scikit-learn
econml
matplotlib
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn econml matplotlib
```

> **Note:** The notebook was run with Python 3.11 on a `pytorch_env` conda kernel.

## How to Run

1. Clone/download this folder.
2. Ensure `ihdp_data.csv` is in the same directory as `notebook.ipynb`.
3. Install the requirements listed above.
4. Open and run `notebook.ipynb` top-to-bottom in Jupyter or VS Code.

## Methods

### Step 1: Data Exploration (EDA)

Loaded `ihdp_data.csv`, inspected shape, dtypes, and treatment balance. Computed `true_ite = mu1 - mu0` and attached it to the dataframe for later validation.

### Step 2: Naive Estimate (Biased Baseline)

Computed the naive Average Treatment Effect (ATE) by directly comparing mean outcomes of treated and control groups:

```
naive_ATE = mean(Y | T=1) − mean(Y | T=0)
```

Compared against `true_ATE = mean(true_ite)` to quantify confounding bias.

### Step 3: Propensity Score Estimation

Fitted a `LogisticRegression` (max_iter=1000) on the 25 covariates to estimate the **propensity score** — the probability of receiving treatment given covariates — for each individual.

Visualized propensity score distributions for treated and control groups to inspect the **overlap (positivity) assumption**.

### Step 4: Propensity Score Matching

Applied **1-nearest-neighbor matching** on the propensity score using `sklearn.neighbors.NearestNeighbors`: for each treated unit, the closest control unit was found. The matched ATE was computed as the mean difference in factual outcomes between matched pairs.

Compared matched ATE to naive ATE and true ATE to confirm that correcting for confounding reduces bias.

### Step 5: Causal Forest (Individual Treatment Effects)

Used **EconML's `CausalForestDML`** with:
- `model_y`: `RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)`
- `model_t`: `LogisticRegression(max_iter=1000)`
- `discrete_treatment=True`

Fitted on the full dataset and estimated individual-level treatment effects (ITEs) via `.effect(X)`.

Computed the Causal Forest ATE (mean of estimated ITEs) and compared it against true ATE.

### Step 6: ITE Validation

- **Scatter plot**: estimated ITEs vs. true ITEs with an ideal diagonal reference line.
- **ITE RMSE**: `sqrt(MSE(true_ite, estimated_ite))` — lower is better.
- **Pearson Correlation**: between estimated and true ITEs — higher is better.

### Step 7: ATE Comparison Table

Summarized all three methods with ATE and absolute bias relative to `true_ATE`:

| Method | ATE | Bias (ATE − True ATE) |
|---|---|---|
| Naive | computed in notebook | computed in notebook |
| Propensity Matching | computed in notebook | computed in notebook |
| Causal Forest | computed in notebook | computed in notebook |

> Actual numerical values are printed in the notebook output cells.

## Key Observations

- **Naive ATE is biased** due to confounding: treated and untreated groups differ systematically in their 25 covariates, so a raw mean comparison does not reflect the true causal effect.
- **Propensity score matching reduces bias** by comparing individuals with similar probabilities of being treated — effectively simulating what a randomized experiment would look like. 1-NN matching is imperfect, so some residual bias remains.
- **The Causal Forest achieves the lowest ATE bias** and additionally produces individual-level treatment effect (ITE) estimates, enabling heterogeneous treatment effect analysis.
- **ITE scatter plot** shows the Causal Forest's ability to capture treatment effect heterogeneity: points cluster around the diagonal when estimation is accurate.
- **ITE RMSE and Pearson correlation** quantify how well individual-level predictions match the known ground truth — metrics only available because `mu0`/`mu1` are simulated.
- This task illustrates the central challenge of causal inference: estimating the **counterfactual** (what would have happened under the treatment not taken), which is never directly observed in real data.

## Conclusion

Using the IHDP semi-synthetic benchmark, three approaches were applied to estimate the Average Treatment Effect:

1. **Naive comparison** — fast but biased due to confounding.
2. **Propensity Score Matching** — reduces confounding bias by matching on treatment probability.
3. **CausalForestDML (EconML)** — the most accurate ATE estimate, plus individual-level ITE estimation with validation against known ground truth.

The progression from naive → matched → causal forest demonstrates how increasingly sophisticated causal methods reduce bias and provide richer insights into treatment effect heterogeneity.
