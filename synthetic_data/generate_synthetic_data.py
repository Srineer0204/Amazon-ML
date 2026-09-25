import pandas as pd
import numpy as np
import os

def generate_synthetic_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Source 1 Data
    s1_data = [
        ["S1-001", "Amazon Inc.", "410 Terry Ave N, Seattle, WA", "US"],
        ["S1-002", "Flipkart Pvt Ltd", "Bengaluru, Karnataka", "India"],
        ["S1-003", "Walmart Stores", "Bentonville, Arkansas", "US"],
        ["S1-004", "Target Corp", "Minneapolis, MN", "US"],
    ]
    pd.DataFrame(s1_data, columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
        os.path.join(output_dir, "train_source1.tsv"), sep="\t", index=False
    )
    
    # Source 2 Data
    s2_data = [
        ["S2-101", "Amazon", "Seattle WA", "US"],
        ["S2-102", "Flipkart", "Bangalore", "India"],
        ["S2-103", "Target Corporation", "Minneapolis", "US"],
    ]
    pd.DataFrame(s2_data, columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
        os.path.join(output_dir, "train_source2.tsv"), sep="\t", index=False
    )
    
    # Source 3 Data
    s3_data = [
        ["S3-201", "Amazon.com", "410 Terry Ave North", "US"],
        ["S3-202", "Walmart", "Bentonville, AR", "US"],
    ]
    pd.DataFrame(s3_data, columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
        os.path.join(output_dir, "train_source3.tsv"), sep="\t", index=False
    )
    
    # Ground Truth
    gt_data = [
        ["S1-001", "S2-101,S3-201"],
        ["S1-002", "S2-102"],
        ["S1-003", "S3-202"],
        ["S1-004", "S2-103"],
    ]
    pd.DataFrame(gt_data, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        os.path.join(output_dir, "train_ground_truth.tsv"), sep="\t", index=False
    )
    
    print("Synthetic data generated successfully.")

if __name__ == "__main__":
    generate_synthetic_data(os.path.join(os.path.dirname(__file__), "data"))
