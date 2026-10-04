# Part C: Image-Based Product Matching

## Objective

Investigate whether visual information from product images can be used to identify matching products using pretrained CNN embeddings.

## Pipeline

Product Image -> Pretrained CNN -> L2-Normalized Embedding -> Cosine Similarity -> Threshold -> Match / No Match

## Dataset

34,250 training images from the Shopee Product Matching dataset.

## Pair Construction

- One positive pair per unique label group (two randomly chosen images from the same group).
- Same number of negative pairs (two randomly chosen images from different groups).

## Experiment 1: ResNet50 (Baseline)

- ResNet50 pretrained on ImageNet, with the top classification layer removed, global average pooling applied.
- Images resized to 224x224, preprocessed with ResNet50's expected normalization.
- Inference run in batches of 64 using tf.data pipeline.
- Produces 2048-dimensional embeddings per image.
- Embeddings L2-normalized before similarity computation.
- Cosine similarity computed between pairs.
- Threshold sweep from 0.30 to 0.95 (step 0.01) to find the best F1 threshold.

Metrics at best threshold: Precision, Recall, F1.

Embeddings saved to esnet50_embeddings.npy for reuse.

## Experiment 2: EfficientNet-B0

- EfficientNet-B0 pretrained on ImageNet, global average pooling, no top layer.
- Same 224x224 resizing, EfficientNet-specific preprocessing.
- Produces 1280-dimensional embeddings.
- Same threshold sweep and evaluation protocol as Experiment 1.

## Results Comparison Table

| Experiment  | Model           | Embedding Dim | Similarity | Threshold | Precision | Recall | F1 |
|-------------|-----------------|---------------|------------|-----------|-----------|--------|----|
| Baseline    | ResNet50        | 2048          | Cosine     | best      | ...       | ...    | ...|
| Experiment 1| EfficientNet-B0 | 1280          | Cosine     | best      | ...       | ...    | ...|

## Nearest-Neighbor Analysis

For a query image, the top-5 most similar images by cosine similarity were retrieved from the full embedding matrix. The matched images were displayed with their similarity scores and product groups to visually verify whether nearest neighbors belong to the same product.

## Error Analysis

- False positives: image pairs predicted as matching but from different label groups. Visualized to identify common patterns (similar backgrounds, similar product categories, similar colors).
- False negatives: image pairs from the same group predicted as not matching. Visualized to identify where the CNN representation failed (different angles, crops, backgrounds).

## Computational Considerations

- Embedding extraction requires passing all 34,250 images through a CNN, which is computationally expensive.
- Comparing all pairs would require approximately 587 million comparisons; sampling was used for evaluation.
- Embeddings are saved to disk and reused rather than recomputed.

## Key Observations

- Pretrained CNN embeddings provide useful visual representations for product matching without training a model from scratch.
- Cosine similarity on L2-normalized embeddings is a natural choice for comparing embedding vectors.
- Image matching struggles when the same product appears with very different backgrounds, cropping, or orientation.
- Visually similar but distinct products (same category, different model) cause false positives.
- Image-based matching complements text-based matching because they fail in different situations.
