import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_dataframe
from src.blocking import generate_candidates_tfidf
from src.features import generate_features_for_candidates

def load_and_sample_data(base_path: str = 'dataset/train', sample_frac: float = 0.01):
    """Load a small sample of the training data for quick testing."""
    print(f"Loading data from {base_path} with sample fraction {sample_frac}...")
    
    s1 = pd.read_csv(os.path.join(base_path, 'train_source1.tsv'), sep='\t', dtype=str)
    s2 = pd.read_csv(os.path.join(base_path, 'train_source2.tsv'), sep='\t', dtype=str)
    s3 = pd.read_csv(os.path.join(base_path, 'train_source3.tsv'), sep='\t', dtype=str)
    gt = pd.read_csv(os.path.join(base_path, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    
    if sample_frac < 1.0:
        # Sample S1 and keep only relevant GT
        s1 = s1.sample(frac=sample_frac, random_state=42)
        gt = gt[gt['source1_entity_id'].isin(s1['entity_id'])]
        
        # S2 and S3 sampling is tricky because we might lose true matches. 
        # For a true test pipeline, we sample S2 and S3, but ensuring we keep matches is hard without 
        # leaking info. For now, just sample them randomly.
        s2 = s2.sample(frac=sample_frac, random_state=42)
        s3 = s3.sample(frac=sample_frac, random_state=42)
        
    s23 = pd.concat([s2, s3], ignore_index=True)
    return s1, s23, gt

def prepare_training_labels(features_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Assign binary labels to the candidate pairs based on ground truth."""
    # Convert GT to a long format: source1_entity_id, matched_entity_id
    gt_records = []
    for _, row in gt_df.dropna(subset=['matched_entity_ids']).iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        for match in matches:
            if match.strip():
                gt_records.append({'source1_entity_id': s1_id, 'candidate_entity_id': match.strip(), 'label': 1})
                
    gt_long = pd.DataFrame(gt_records)
    
    # Merge with features
    if gt_long.empty:
        features_df['label'] = 0
    else:
        features_df = features_df.merge(
            gt_long, 
            on=['source1_entity_id', 'candidate_entity_id'], 
            how='left'
        )
        features_df['label'] = features_df['label'].fillna(0).astype(int)
        
    return features_df

def run_training_pipeline():
    # 1. Load Data
    # Use full dataset for final training
    s1, s23, gt = load_and_sample_data(sample_frac=1.0)
    print(f"Sampled S1: {len(s1)}, S2+S3: {len(s23)}")
    
    # 2. Preprocess
    print("Preprocessing data...")
    s1_prep = preprocess_dataframe(s1)
    s23_prep = preprocess_dataframe(s23)
    
    # 3. Blocking
    print("Generating candidates...")
    candidates = generate_candidates_tfidf(s1_prep, s23_prep, text_column='norm_name', threshold=0.6)
    print(f"Generated {len(candidates)} candidate pairs.")
    
    if len(candidates) == 0:
        print("No candidates generated. Try lowering the TF-IDF threshold or increasing sample size.")
        return
        
    # 4. Feature Engineering
    print("Extracting features...")
    features_df = generate_features_for_candidates(candidates, s1_prep, s23_prep)
    
    # 5. Labeling
    features_df = prepare_training_labels(features_df, gt)
    print(f"Label distribution:\n{features_df['label'].value_counts()}")
    
    # 6. Train Model
    feature_cols = [col for col in features_df.columns if col not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    
    X = features_df[feature_cols]
    y = features_df['label']
    
    # Only train if we have both classes
    if len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = HistGradientBoostingClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        # Get prediction probabilities for the positive class
        probs = clf.predict_proba(X_test)[:, 1]
        
        print("\n--- Tuning Decision Threshold ---")
        best_threshold = 0.5
        best_f05 = 0.0
        best_metrics = {}
        
        # Search for the best threshold between 0.3 and 0.95
        for threshold in np.arange(0.3, 0.96, 0.05):
            preds = (probs >= threshold).astype(int)
            
            # Avoid division by zero if there are no positive predictions
            if sum(preds) == 0:
                continue
                
            p = precision_score(y_test, preds, zero_division=0)
            r = recall_score(y_test, preds, zero_division=0)
            f05 = fbeta_score(y_test, preds, beta=0.5, zero_division=0)
            
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = threshold
                best_metrics = {'precision': p, 'recall': r, 'f05': f05}
                
        print(f"Optimal Threshold found: {best_threshold:.2f}")
        print(f"Optimized Precision: {best_metrics['precision']:.4f}")
        print(f"Optimized Recall: {best_metrics['recall']:.4f}")
        print(f"Optimized F0.5 Score: {best_metrics['f05']:.4f}")
        
        # Save the model and threshold
        import joblib
        import os
        os.makedirs('models', exist_ok=True)
        
        # Retrain on full available data (X, y) instead of just X_train to maximize learning
        clf_full = HistGradientBoostingClassifier(random_state=42)
        clf_full.fit(X, y)
        
        model_data = {
            'model': clf_full,
            'threshold': best_threshold,
            'features': feature_cols
        }
        joblib.dump(model_data, 'models/er_model.pkl')
        print("Model and optimal threshold saved to models/er_model.pkl")
    else:
        print("Not enough variation in labels to train the model (all candidates might be negative).")

if __name__ == "__main__":
    run_training_pipeline()
