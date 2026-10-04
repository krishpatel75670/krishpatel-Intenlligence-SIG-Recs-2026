# Intelligence SIG Recruitment Tasks 2026 — Krish Patel

This repository contains my submissions for the Intelligence SIG recruitment tasks, Web Club NITK 2026.

---

## About Me

**Name:** Krish Patel
**Repository:** krishpatel-Intelligence-SIG-Recs-2026

---

## Tasks Attempted

### Mandatory Tasks

All three mandatory tasks have been completed.

---

#### 1. Recommender Systems: From Collaborative Filtering to DLRM

**Location:** mandatory-tasks/recommender-systems-cf-to-dlrm

This task involved building a progression of recommendation systems, starting from classical collaborative filtering approaches and advancing to a modern Deep Learning Recommendation Model (DLRM)-style architecture.

**What was done:**

- **Subtask 1 — Collaborative Filtering:** Implemented and compared memory-based (user-user and item-item) collaborative filtering with model-based matrix factorization. Evaluated on a user-item interaction dataset using standard ranking and accuracy metrics.

- **Subtask 2 — Neural CTR Model:** Moved beyond linear assumptions to train a neural network that learns nonlinear feature combinations for click-through rate prediction. The dataset contains 13 numeric features, 26 categorical features, and a binary click label.

- **Subtask 3 — DLRM:** Assembled dense features, sparse categorical embeddings, and explicit pairwise feature interactions in a DLRM-style architecture. Compared against the simpler neural baseline to quantify what the architectural progression gained.

**Metrics reported:** ROC-AUC, PR-AUC, Accuracy, Log Loss, F1.

---

#### 2. Seq2Seq Grounded Dialogue Generation

**Location:** mandatory-tasks/seq2seq-grounded-dialogue-generation

This task involved building sequence-to-sequence NLP systems from scratch — starting from a basic encoder-decoder for machine translation and progressively extending to a document-grounded dialogue generator in Hinglish (Hindi-English code-mixed text). No pretrained transformer fine-tuning or packaged seq2seq pipelines were used.

**What was done:**

- **Subtask 1 — Machine Translation (English to French):** Built an LSTM-based encoder-decoder with attention. Implemented and compared Bahdanau and Luong attention mechanisms. Evaluated using BLEU score on the provided EN-FR dataset.

- **Subtask 2 — Extended Seq2Seq:** Extended the translation model with deeper architectures and different decoding strategies (greedy vs. beam search). Analyzed the effect of attention and decoding choices on output quality.

- **Subtask 3 — Grounded Dialogue Generation:** Adapted the architecture for multi-turn, document-grounded dialogue generation in Hinglish. The model conditions simultaneously on the conversation history and a retrieved document. Evaluated generation quality and documented the conditioning pipeline.

---

#### 3. Shopee Product Matching

**Location:** mandatory-tasks/shopee-product-matching

This task involved building a product matching system for the Shopee Product Matching dataset, where the goal is to identify e-commerce listings that correspond to the same underlying product despite having different titles, images, sellers, and backgrounds.

**What was done:**

- **Part A — Dataset Exploration:** Analyzed the dataset structure, label group distributions, image and title diversity, and identified challenges such as noisy titles, varied image orientations, and class imbalance.

- **Part B — Text-Based Matching:** Built a text similarity pipeline using TF-IDF, embedding-based methods, and threshold tuning on cosine similarity scores to group matching products by title.

- **Part C — Image-Based Matching:** Extracted visual features using CNN-based embeddings and applied similarity search to identify visually matching product listings.

- **Finale — Multimodal Matching:** Combined text and image similarity signals into a unified matching pipeline, experimenting with different fusion strategies and evaluating the gain from multimodal information over unimodal baselines.

---

### Non-Mandatory Tasks

One non-mandatory task was completed.

---

#### 4. Causal Inference

**Location:** non-mandatory-tasks/causal-inference

This task involved moving beyond correlation-based machine learning to estimate causal treatment effects using the IHDP (Infant Health and Development Program) semi-synthetic benchmark dataset. IHDP provides real covariates with simulated outcomes, making ground-truth treatment effects available for validation.

**What was done:**

- **Naive Estimate:** Computed the naive Average Treatment Effect (ATE) by directly comparing mean outcomes between treated and untreated groups to establish a biased baseline.

- **Propensity Score Methods:** Implemented inverse propensity weighting (IPW) and propensity score matching to correct for confounding. Re-estimated ATE and compared against the naive baseline.

- **Causal Forest (Heterogeneous Treatment Effects):** Used the EconML or CausalML library to implement a causal forest, estimating Individual Treatment Effects (ITEs) across the population rather than just the average.

- **Validation and Overlap Analysis:** Plotted estimated ITEs against ground-truth ITEs to quantify estimation quality. Visualized propensity score distributions for treated vs. untreated groups to check the overlap/positivity assumption and discussed the effect of poor overlap regions on estimates.

**Key insight:** The gap between the naive estimate and the corrected causal estimates demonstrates the practical impact of confounding, and the ITE estimates show how treatment effects vary meaningfully across individuals.

---

## Repository Structure

    mandatory-tasks/
        recommender-systems-cf-to-dlrm/
            RecommendationTasks/
                01_CollaborativeFiltering/   - CF vs Matrix Factorization
                02_NeuralCTR/               - Neural click-through rate model
                03_DLRM/                    - DLRM-style architecture
            datasets/                       - Advertising CTR dataset

        seq2seq-grounded-dialogue-generation/
            subtask1/                       - EN-FR translation with attention
            subtask2/                       - Extended seq2seq, beam search
            subtask3/                       - Hinglish grounded dialogue generation
            en_fr_train.csv / val / test    - Translation dataset

        shopee-product-matching/
            Part_A/                         - Dataset exploration
            Part_B/                         - Text-based matching
            Part_C/                         - Image-based matching
            Final/                          - Multimodal matching

    non-mandatory-tasks/
        causal-inference/
            notebook.ipynb                  - Full causal inference pipeline
            ihdp_data.csv                   - IHDP benchmark dataset

---

## Notes

- All code is original. External libraries, papers, and references used are cited within each notebook.
- No pretrained transformer weights were fine-tuned in the seq2seq task.
- Each notebook includes exploratory analysis, training details, results, and a short discussion of findings and limitations.
