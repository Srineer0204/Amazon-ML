import pandas as pd
from rapidfuzz import fuzz, distance
import numpy as np

def compute_string_similarities(str1: str, str2: str) -> dict:
    """
    Compute various string similarity metrics between two strings.
    Handles None or NaN values gracefully.
    """
    if pd.isna(str1) or pd.isna(str2) or not isinstance(str1, str) or not isinstance(str2, str):
        return {
            'levenshtein_ratio': 0.0,
            'jaro_winkler': 0.0,
            'token_set_ratio': 0.0,
            'exact_match': 0
        }
        
    return {
        # fuzz.ratio returns 0-100, we normalize to 0-1
        'levenshtein_ratio': fuzz.ratio(str1, str2) / 100.0,
        'jaro_winkler': distance.JaroWinkler.normalized_similarity(str1, str2),
        'token_set_ratio': fuzz.token_set_ratio(str1, str2) / 100.0,
        'exact_match': 1 if str1 == str2 else 0
    }

def generate_features_for_candidates(candidates_df: pd.DataFrame, 
                                     source1_df: pd.DataFrame, 
                                     source23_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of candidate pairs, join the original data and compute features.
    
    Args:
        candidates_df: DataFrame with 'source1_entity_id' and 'candidate_entity_id'
        source1_df: Preprocessed Source 1 data
        source23_df: Preprocessed Source 2 & 3 data
        
    Returns:
        DataFrame with features ready for ML modeling.
    """
    # Merge S1 data
    features_df = candidates_df.merge(
        source1_df[['entity_id', 'norm_name', 'norm_address', 'norm_country']],
        left_on='source1_entity_id', 
        right_on='entity_id',
        how='left'
    ).rename(columns={
        'norm_name': 's1_name', 
        'norm_address': 's1_address', 
        'norm_country': 's1_country'
    }).drop(columns=['entity_id'])
    
    # Merge S23 data
    features_df = features_df.merge(
        source23_df[['entity_id', 'norm_name', 'norm_address', 'norm_country']],
        left_on='candidate_entity_id', 
        right_on='entity_id',
        how='left'
    ).rename(columns={
        'norm_name': 's23_name', 
        'norm_address': 's23_address', 
        'norm_country': 's23_country'
    }).drop(columns=['entity_id'])
    
    # Compute features
    print("Computing name similarities...")
    name_features = features_df.apply(
        lambda row: compute_string_similarities(row['s1_name'], row['s23_name']), 
        axis=1, result_type='expand'
    ).add_prefix('name_')
    
    print("Computing address similarities...")
    address_features = features_df.apply(
        lambda row: compute_string_similarities(row['s1_address'], row['s23_address']), 
        axis=1, result_type='expand'
    ).add_prefix('address_')
    
    # Country feature
    features_df['country_match'] = (features_df['s1_country'] == features_df['s23_country']).astype(int)
    
    # Combine all features
    final_features_df = pd.concat([features_df, name_features, address_features], axis=1)
    
    # Drop raw string columns as the model only needs the numerical features
    cols_to_drop = ['s1_name', 's1_address', 's1_country', 's23_name', 's23_address', 's23_country']
    final_features_df = final_features_df.drop(columns=cols_to_drop)
    
    return final_features_df
