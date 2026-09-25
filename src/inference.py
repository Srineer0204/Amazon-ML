import pandas as pd
import numpy as np
import joblib
import os
import argparse
import sys

# Ensure Python can find the 'src' module when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_dataframe
from src.blocking import generate_candidates_tfidf, format_candidates_for_output
from src.features import generate_features_for_candidates

def load_test_data(base_path: str = 'dataset/test', sample_frac: float = 1.0):
    """Load the test data."""
    print(f"Loading test data from {base_path}...")
    s1 = pd.read_csv(os.path.join(base_path, 'test_source1.tsv'), sep='\t', dtype=str)
    s2 = pd.read_csv(os.path.join(base_path, 'test_source2.tsv'), sep='\t', dtype=str)
    s3 = pd.read_csv(os.path.join(base_path, 'test_source3.tsv'), sep='\t', dtype=str)
    
    if sample_frac < 1.0:
        s1 = s1.sample(frac=sample_frac, random_state=42)
        s2 = s2.sample(frac=sample_frac, random_state=42)
        s3 = s3.sample(frac=sample_frac, random_state=42)
        
    s23 = pd.concat([s2, s3], ignore_index=True)
    return s1, s23

def run_inference(sample_frac: float = 1.0):
    os.makedirs('output', exist_ok=True)
    
    # 1. Load Model
    model_path = 'models/er_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
        
    print("Loading trained model...")
    model_data = joblib.load(model_path)
    clf = model_data['model']
    threshold = model_data['threshold']
    feature_cols = model_data['features']
    print(f"Using threshold: {threshold}")
    
    # 2. Load Data
    s1, s23 = load_test_data(sample_frac=sample_frac)
    
    # 3. Preprocess
    print("Preprocessing test data...")
    s1_prep = preprocess_dataframe(s1)
    s23_prep = preprocess_dataframe(s23)
    
    # 4. Candidate Generation (Blocking)
    print("Generating candidate pairs...")
    # NOTE: In a real run, threshold might need to be lower to catch all matches. 0.5 is a balance.
    candidates = generate_candidates_tfidf(s1_prep, s23_prep, text_column='norm_name', threshold=0.5)
    
    # Save candidate pairs output BEFORE filtering!
    print("Saving candidate_pairs.tsv...")
    candidate_output = format_candidates_for_output(candidates)
    
    # Ensure all S1 entities are in the output (singletons must be empty strings)
    all_s1 = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    candidate_output = all_s1.merge(candidate_output, on='source1_entity_id', how='left')
    candidate_output['candidate_entity_ids'] = candidate_output['candidate_entity_ids'].fillna('')
    candidate_output.to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
    
    if candidates.empty:
        print("No candidates found. Exiting.")
        return
        
    # 5. Feature Engineering
    print("Extracting features for candidates...")
    features_df = generate_features_for_candidates(candidates, s1_prep, s23_prep)
    
    # 6. Predict
    print("Predicting matches...")
    X = features_df[feature_cols]
    probs = clf.predict_proba(X)[:, 1]
    
    # Apply optimal threshold
    features_df['is_match'] = (probs >= threshold).astype(int)
    
    # Filter only positive matches
    matches = features_df[features_df['is_match'] == 1]
    
    # 7. Format Final Output
    print("Saving matching_results.tsv...")
    if matches.empty:
        matching_output = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
    else:
        # Group by S1 and join S23 ids
        matching_output = matches.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(x.dropna().unique())
        ).reset_index()
        matching_output.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    
    # Ensure every single Source 1 entity from the test set is in the final file
    final_results = all_s1.merge(matching_output, on='source1_entity_id', how='left')
    final_results['matched_entity_ids'] = final_results['matched_entity_ids'].fillna('')
    
    # Sort for neatness (optional)
    final_results = final_results.sort_values('source1_entity_id')
    
    final_results.to_csv('output/matching_results.tsv', sep='\t', index=False)
    print("Done! Outputs saved to the 'output/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_frac', type=float, default=1.0, help='Fraction of data to sample')
    args = parser.parse_args()
    run_inference(args.sample_frac)
