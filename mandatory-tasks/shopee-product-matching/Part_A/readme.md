# Part A: Dataset Exploration

## Objective

Understand the Shopee Product Matching dataset before attempting to build a matching system.

## Dataset

Shopee Product Matching dataset from Kaggle. The training CSV was used for the main analysis (it contains label_group representing product identity). The test CSV was inspected to understand available columns.

## Columns

- posting_id: unique listing identifier
- image: image filename for the listing
- image_phash: perceptual hash of the image
- 	itle: text title of the listing
- label_group: product-group identifier (same value = same product, training only)

## What Was Done

### Dataset Statistics

Computed and displayed:
- Number of product listings and unique product groups
- Distribution of product-group sizes (histogram + description)
- Largest and average group size
- Number of unique image filenames and pHash values
- Duplicate image rows and duplicate pHash rows
- Number of unique titles and duplicate title rows
- Title length distribution (average, min, max, histogram)

### Top Product Groups

Identified the 10 largest product groups by listing count. Visualized their sizes with a bar chart.

### Duplicate and Highly Similar Titles

- Identified exact duplicate title rows and displayed examples with their label groups.
- Used character n-gram TF-IDF (3-5 grams) cosine similarity on a sample of 10,000 titles to find highly similar but not identical title pairs.
- Separated these into two categories: similar titles from the same product group (expected), and similar titles from different product groups (potential false positives for text-only matching).

### Same Product, Different Images

Found product groups where the image filename differs across listings. Among these, further filtered for groups where the perceptual hash also differs, showing that the same product identity can have meaningfully different image representations.

### Visual Similarity Across Different Products

Computed Hamming distance between hexadecimal pHash values to find visually similar images from different product groups. These are examples where image similarity alone would produce false positive matches.

### Title Character Analysis

Computed per-listing statistics: title length, fraction of non-ASCII characters, digit count, punctuation count. Visualized non-ASCII character ratio distribution to show multilingual and mixed-script titles.

### Short and Potentially Noisy Listings

Identified listings with the shortest titles to show cases with limited textual information.

## Identified Matching Challenges

1. Noisy product titles: extra words, seller-specific text, punctuation, formatting
2. Multilingual and mixed character sets: titles are not consistently written in one language or script
3. Abbreviations and variants: same product, different wording
4. Visually similar products with different identities: pHash Hamming distance can be low across groups
5. Same product, different images: different orientations, backgrounds, crops
6. Exact or near-duplicate images across different product identities
7. Short titles with minimal information
8. Highly variable group sizes: some groups have many listings, others only a few

## Conclusions

- Titles alone cannot reliably determine product identity because similar titles appear across different groups and different titles appear within the same group.
- Images alone are also insufficient: the same product can appear differently and different products can look visually similar.
- A robust matching system will need to combine textual and visual signals and handle the noise and diversity in both.
