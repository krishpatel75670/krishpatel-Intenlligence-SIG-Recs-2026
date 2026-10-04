# Subtask 2: Attention Mechanism + Decoding Strategies

## Objective

Extend the basic encoder-decoder from Subtask 1 with Bahdanau attention and compare greedy vs beam-search decoding on the same English-French translation task.

## Approach

### Starting Point

The trained basic Seq2Seq model from Subtask 1 was saved as seq2seq_basic.pkl. This notebook loads the checkpoint and reuses the vocabulary, encoder embeddings, encoder LSTMs, decoder LSTMs, and output projection layer.

The attention model is initialized from these pretrained weights, then a new Bahdanau attention layer is added and the entire model is fine-tuned.

### Bahdanau Attention

At each decoder step:
- The decoder's last hidden state is used as the query.
- All encoder output states are the values.
- Additive (Bahdanau) attention computes a score for each encoder position: score = V(tanh(W1(value) + W2(query))).
- A softmax over scores gives attention weights; the context vector is the weighted sum of encoder outputs.
- PAD positions in the source are masked with -1e9 before softmax.
- The context vector is projected and concatenated with the embedding before each decoder RNN step.

### Training the Attention Model

- The encoder and attention-augmented decoder were trained together on the full training data.
- Adam optimizer (lr=1e-3), gradient clipping (norm 1.0).
- Teacher forcing was used during training (input is the previous ground-truth token).
- Early stopping on validation loss (patience=2).
- 3 epochs of fine-tuning after loading the pretrained base weights.

### Decoding

- Greedy decoding: picks the highest-probability token at each step.
- The attention model with greedy decoding was compared against the basic model.

### Evaluation

- BLEU computed on the test set using SacreBLEU.
- Example outputs compared side by side: EN / REFERENCE / ATTENTION output.

### Results

The attention model achieves higher BLEU than the fixed-context basic model because it can focus on relevant source positions at each generation step rather than relying on a single compressed context vector.

### Saved Artifacts

- seq2seq_attention.pkl: full attention model checkpoint (config, vocabulary, encoder weights, decoder weights)
- ttention_encoder.weights.h5: encoder weights at the best validation epoch
- ttention_decoder.weights.pkl: decoder weights at the best validation epoch
