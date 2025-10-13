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

### Collaborative Filtering through Matrix Factorization

<img width="1095" height="597" alt="Collaborative Filtering" src="https://github.com/user-attachments/assets/93bc2e5b-7873-428b-90c9-7e859dacbd86" />

**Figure**: Collaborative filtering through matrix factorization.  
Phase 1 learns latent factors from known interactions (yellow cells).  
Phase 2 uses these factors to predict unknown interactions (grey cells).


### Mathematical Formulation

The recommendation score for a user-item pair is calculated as:

$$
\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i
$$

Where:
- $\hat{r}_{ui}$ is the predicted interaction score for user $u$ and item $i$.
- $\mathbf{p}_u$ is the latent factor vector for user $u$.
- $\mathbf{q}_i$ is the latent factor vector for item $i$.

The optimization objective minimizes the confidence-weighted loss:

$$
\min_{P, Q} \sum_{(u, i) \in \mathcal{R}} c_{ui} (r_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2 + \lambda (\|P\|^2 + \|Q\|^2)
$$

Where:
- $c_{ui}$ is the confidence weight for the interaction $(u, i)$.
- $r_{ui}$ is the observed interaction value.
- $\lambda$ is the regularization parameter.

## Getting Started

Follow these steps to set up and run the project:

### Prerequisites
- Python 3.8 or higher
- Git
- Jupyter Notebook

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mkuangdotcom/Music_Recommendation_System.git
   cd Music_Recommendation_System
   ```
2. Open `User_Centric_Approach_to__Music_Recommendations.ipynb` in Jupyter Notebook.
   ```bash
    jupyter notebook User_Centric_Approach_to__Music_Recommendations.ipynb
    ``` 
3. Run the notebook

## Implementation Highlights

<img width="332" height="118" alt="CSR Matrix" src="https://github.com/user-attachments/assets/b5f9b42d-015a-4542-8de9-3c7b719c0ca0" />

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
