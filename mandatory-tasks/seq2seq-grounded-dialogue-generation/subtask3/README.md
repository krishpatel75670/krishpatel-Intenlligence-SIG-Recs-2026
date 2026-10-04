# Subtask 3: Document-Grounded Dialogue Generation in Hinglish

## Objective

Extend the encoder-decoder architecture from Subtasks 1 and 2 to a multi-turn, document-grounded dialogue generator that generates responses in Hinglish (Hindi-English code-mixed text).

## Dataset

CMU Hinglish DoG (Document-Grounded Conversations) loaded from HuggingFace (estvox/cmu_hinglish_dog):
- Real two-person chat sessions discussing a topic grounded in a Wikipedia document (CMU DoG WikiData)
- Each turn is available in both English and Hinglish (hi_en)
- Includes speaker identity, document section index, session metadata, and a flag for whether the speaker saw the document

## Architecture

### Two-Encoder Design

The model (DoGModel) uses two separate BiGRU encoders with shared embeddings:
1. History encoder: encodes the last 3 turns of dialogue history (marked with speaker tokens <s1>, <s2>, and separated by <sep>)
2. Document encoder: encodes the relevant Wikipedia section (up to 200 tokens)

The decoder is a GRU cell that attends over both encoder outputs using two separate Bahdanau attention layers at every generation step.

### Shared Embeddings

A single embedding matrix (vocabulary size 12,000, dimension 128) is shared between both encoders and the decoder. The embedding matrix is initialized from FastText vectors trained on the training utterances and document tokens, then fine-tuned during training.

### Decoder

At each step:
- Two context vectors are computed: one from the history encoder, one from the document encoder
- These are concatenated with the embedded previous token and fed into the GRU cell
- A mixing layer projects the combined representation before the output projection

### Vocabulary and Preprocessing

- Word-level tokenization with lowercase and URL normalization
- Custom Vocab class that prioritizes document words (seen at least once) and utterance words (seen at least twice)
- Special tokens: <pad>, <unk>, <bos>, <eos>, <sep>, <s1>, <s2>

### Training

- Adam optimizer (lr=1e-3), gradient clipping (norm 1.0)
- Masked cross-entropy loss (ignoring PAD tokens)
- Early stopping (patience=3) on validation loss
- Up to 20 epochs

## Baselines

Two baselines were implemented:

1. Retrieval baseline: returns the training response whose history is most similar to the test history using TF-IDF cosine similarity.
2. Ungrounded seq2seq: same DoGModel architecture but with the document encoder disabled (use_doc=False). Trained identically, so the comparison isolates the effect of document grounding.

## Evaluation

All systems evaluated on the test set using:
- BLEU (SacreBLEU)
- ROUGE-1 and ROUGE-L
- Distinct-1 and Distinct-2 (lexical diversity)
- Document overlap on grounded turns: fraction of content words in the reply that appear in the document
- ROUGE-L broken down by Hinglish mix bucket: mostly_english, code_mixed, mostly_hindi

Qualitative samples compared HIST / REF / RETRIEVAL / UNGROUNDED / GROUNDED responses.

## Key Design Decisions

- Speaker tokens in the history signal who speaks next, which matters for generating the correct reply register
- Document words are kept in the vocabulary even with low frequency to avoid replacing named entities with <unk>
- The positivity assumption check (doc overlap on grounded vs ungrounded speakers) validates whether the document actually influences grounded speakers' replies
