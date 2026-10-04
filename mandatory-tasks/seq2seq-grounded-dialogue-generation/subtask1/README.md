# Subtask 1: Basic Sequence-to-Sequence Translation (English to French)

## Objective

Build the foundational LSTM encoder-decoder architecture from scratch and evaluate it on English-to-French translation.

## Dataset

The provided English-French parallel corpus was used:
- en_fr_train.csv
- en_fr_val.csv
- en_fr_test.csv

Duplicate rows were removed from the training split. Sentences longer than 40 tokens on either side were filtered out from training only; the validation and test sets were kept intact for honest evaluation.

## Approach

### Tokenization and Vocabulary

- Word-level tokenization with lower-casing and punctuation splitting (e.g., l'annee becomes l ' annee).
- Separate vocabularies for English and French, each capped at 16,000 tokens.
- Words appearing fewer than 2 times become <unk>.
- Special tokens: <pad>, <sos>, <eos>, <unk>.

### Architecture

A 2-layer LSTM encoder-decoder without attention:

`
source tokens -> Embedding(256) -> LSTM(512) x2 -> final (h, c)
                                                         |
<sos> y1 y2 ...  -> Embedding(256) -> LSTM(512) x2, init=(h,c) -> Linear -> P(next word)
`

- Encoder compresses the entire source sentence into a fixed-size context vector.
- Decoder generates the target sequence token by token from this fixed context.
- Dropout (0.3) applied on embeddings and between LSTM layers.

### Training

- Adam optimizer (lr=1e-3, epsilon=1e-8).
- Gradient clipping (max norm 1.0).
- Cross-entropy loss averaged over non-padding target tokens.
- Learning rate scheduler: ReduceLROnPlateau on validation loss (factor=0.5, patience=1).
- Early stopping with patience=3 on validation loss.
- Up to 12 epochs.

### Inference

Greedy decoding: the model encodes the source once, then repeatedly picks the most probable next token until <eos> or the length cap (80 tokens).

### Evaluation

- Corpus-level BLEU computed on the full test split using SacreBLEU.
- Failure mode analysis by slicing the test set by source length and number of unknown words.
- Repetition rate (3-gram repeats) measured in generated outputs.

## Key Observations

- The model performs well on short sentences but BLEU drops significantly for longer sentences (40+ tokens) where the fixed context vector becomes a bottleneck.
- Sentences with unknown source words also show lower BLEU.
- A non-trivial percentage of outputs contain repeated 3-grams, which motivates adding attention in Subtask 2.
- This basic model establishes the baseline that the attention model in Subtask 2 must improve on.
