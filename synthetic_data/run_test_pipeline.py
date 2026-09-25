import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data import load_data, validate_columns, handle_missing_values, report_record_counts, report_country_distribution, inspect_ground_truth, load_ground_truth
from generate_synthetic_data import generate_synthetic_data

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # 1. Generate synthetic data
    generate_synthetic_data(data_dir)
    
    print("\n--- Testing Data Pipeline ---")
    # 2. Load and validate Source 1
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s1_df = load_data(s1_path)
    validate_columns(s1_df, ["entity_id", "business_name", "business_address", "country"])
    s1_df = handle_missing_values(s1_df)
    report_record_counts(s1_df, "Source 1")
    report_country_distribution(s1_df, "Source 1")
    
    # 3. Load and validate Source 2
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s2_df = load_data(s2_path)
    validate_columns(s2_df, ["entity_id", "business_name", "business_address", "country"])
    s2_df = handle_missing_values(s2_df)
    report_record_counts(s2_df, "Source 2")
    
    # 4. Load and validate Ground Truth
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    gt_df = load_ground_truth(gt_path)
    inspect_ground_truth(gt_df)
    
    print("\nAll data pipeline tests passed successfully!")

if __name__ == "__main__":
    main()
