# Amazon ML Challenge 2026: Business Entity Resolution
## Methodology Document

### 1. Methodology Used
Our solution to the Business Entity Resolution challenge is a two-stage machine learning pipeline consisting of a **Blocking/Candidate Generation** phase followed by a **Pairwise Classification** phase. 
Because comparing every Source 1 record against all Source 2 and 3 records is computationally infeasible ($O(N \times M)$), we first reduce the search space using TF-IDF and cosine similarity. Once a shortlist of plausible candidates is generated, we extract advanced string similarity features and train a `HistGradientBoostingClassifier` to score the pairs. Finally, we optimize the decision threshold to heavily favor Precision, maximizing the target $F_{0.5}$ metric.

### 2. Candidate Generation/Blocking Strategy
To generate `candidate_pairs.tsv` efficiently:
- **Text Normalization:** We preprocess all business names by lowercasing, removing punctuation, and expanding common legal suffixes (e.g., "corp" to "corporation", "pvt" to "private").
- **TF-IDF Vectorization:** We compute character-level n-grams (sizes 2 to 4) using `TfidfVectorizer` on the normalized business names.
- **Cosine Similarity:** We calculate the cosine similarity between Source 1 entities and Source 2/3 entities. Pairs with a similarity score above a strict threshold are kept as candidates. This ensures high recall while drastically reducing the number of pairs passed to the ML model.

### 3. Model Architecture and Feature Engineering
For the candidate pairs generated in the blocking step, we extract the following features to feed into the model:
- **Name Features:** Levenshtein Ratio, Jaro-Winkler Similarity, and Token Set Ratio between the normalized business names using the `RapidFuzz` library.
- **Address Features:** We normalize addresses by expanding common street abbreviations (e.g., "st" to "street", "rd" to "road"). We then compute the same string similarity metrics (Levenshtein, Jaro-Winkler, Token Set) on the addresses.
- **Categorical Features:** An exact match boolean feature for the `country` column.

**Model:** We use a `HistGradientBoostingClassifier` (via `scikit-learn`). This tree-based model is highly efficient, naturally handles tabular data, and performs excellently on bounded numerical features (like our 0-1 similarity scores). It is well within the < 8 Billion parameter constraint.

### 4. Post-processing and Optimization
Because the evaluation metric is $F_{0.5}$ (which penalizes false positives twice as much as false negatives), standard 0.5 probability thresholds perform poorly. 
During training, we iterate through various decision thresholds (from 0.3 to 0.95) using the validation set to find the optimal cut-off that maximizes the $F_{0.5}$ score. The model ultimately uses a very strict confidence threshold (e.g., >0.90) to ensure high precision before predicting a final match. Singletons are correctly handled and left empty in the final output.
