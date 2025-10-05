# Music Recommendation System: Collaborative Filtering with ALS

Implementation of personalized music recommendations using Alternating Least Squares (ALS) matrix factorization on the Last.fm 360K dataset.

## Problem

Music streaming platforms struggle with three issues:
- Popularity bias: Popular items dominate recommendations
- Cold-start: New artists get zero visibility
- No personalization: Everyone sees the same recommendations

## Solution

Compare two approaches:
- Baseline: Recommend top popular artists to everyone
- ALS: Learn user preferences through collaborative filtering

## Results

| Metric | Baseline | ALS | Improvement |
|--------|----------|-----|-------------|
| Precision@10 | 2.8% | 13.7% | 4.8x |
| Recall@10 | 2.2% | 10.5% | 4.8x |
| Coverage | 10 artists | 4,352 artists | 435x |

ALS breaks the popularity bias cycle by recommending from 6.5% of the catalog instead of 0.01%.

## Dataset

Last.fm 360K: 17.5M listening records from 358K users across 160K artists. Filtered to top 20K active users for efficiency, resulting in 1.36M interactions (99.9% sparse matrix).

## Technical Approach

Matrix factorization decomposes sparse user-item matrix into latent factors:
- User factors: 20,000 users x 100 dimensions
- Item factors: 66,959 artists x 100 dimensions
- Prediction: user_i · artist_j = recommendation score

Confidence-weighted loss handles implicit feedback (play counts):
- High play counts = high confidence in preference
- Zero plays = low confidence (could be dislike OR unawareness)

Hyperparameters tuned via random search over 12 trials.

## Usage

Open notebook in Google Colab. Self-contained with automatic setup:
- Dependencies installed via pip
- Dataset auto-downloaded from Zenodo
- Runs end-to-end without manual configuration

## Implementation Highlights

- Sparse matrix storage (CSR format) for memory efficiency
- Per-user train/test split (80/20) to simulate future predictions
- Evaluation across multiple metrics: Precision, Recall, NDCG, Coverage
- Convergence analysis shows diminishing returns after 10 iterations

## Limitations

- Cold-start: Cannot recommend for new users/artists
- Missing features: Demographics (age, country) not incorporated
- Static: No temporal modeling of changing preferences
- Sparsity: 99.9% missing data limits learning

## Future Work

- Factorization Machines to use demographic features
- Neural collaborative filtering for non-linear patterns
- Temporal dynamics for evolving user tastes

## Files

- `User_Centric_Approach_to__Music_Recommendations.ipynb` - Complete implementation with analysis
- `FinalReport_26130714.pdf` - Detailed technical documentation

## References

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.

Dataset: https://zenodo.org/record/6090214
