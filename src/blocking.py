import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_candidates_tfidf(source1_df: pd.DataFrame, source23_df: pd.DataFrame, 
                              text_column: str = 'norm_name', 
                              threshold: float = 0.5) -> pd.DataFrame:
    """
    Generate candidate pairs using TF-IDF cosine similarity.
    
    Args:
        source1_df: DataFrame containing Source 1 records.
        source23_df: DataFrame containing Source 2 and Source 3 records.
        text_column: The column to use for TF-IDF (e.g., 'norm_name' or a combined text column).
        threshold: Minimum cosine similarity score to consider as a candidate.
        
    Returns:
        A DataFrame with candidate pairs.
    """
    # Ensure no missing values in the text column
    s1_texts = source1_df[text_column].fillna("").tolist()
    s23_texts = source23_df[text_column].fillna("").tolist()
    
    # Fit TF-IDF on all texts
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    vectorizer.fit(s1_texts + s23_texts)
    
    s1_tfidf = vectorizer.transform(s1_texts)
    s23_tfidf = vectorizer.transform(s23_texts)
    
    # Compute cosine similarity
    # similarity_matrix shape: (len(s1), len(s23))
    similarity_matrix = cosine_similarity(s1_tfidf, s23_tfidf)
    
    # Find pairs above threshold
    s1_indices, s23_indices = np.where(similarity_matrix >= threshold)
    
    candidates = []
    for s1_idx, s23_idx in zip(s1_indices, s23_indices):
        candidates.append({
            'source1_entity_id': source1_df.iloc[s1_idx]['entity_id'],
            'candidate_entity_id': source23_df.iloc[s23_idx]['entity_id'],
            'similarity_score': similarity_matrix[s1_idx, s23_idx]
        })
        
    return pd.DataFrame(candidates)

def format_candidates_for_output(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the candidates dataframe to the required candidate_pairs.tsv format.
    
    Required format:
    source1_entity_id    candidate_entity_ids (comma-separated)
    """
    if candidates_df.empty:
         return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_ids'])
         
    grouped = candidates_df.groupby('source1_entity_id')['candidate_entity_id'].apply(
        lambda x: ','.join(x.dropna().unique())
    ).reset_index()
    
    grouped.rename(columns={'candidate_entity_id': 'candidate_entity_ids'}, inplace=True)
    return grouped
