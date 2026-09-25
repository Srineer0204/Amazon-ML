import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load TSV file safely."""
    try:
        return pd.read_csv(filepath, sep='\t', dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")

def validate_columns(df: pd.DataFrame, expected_columns: list) -> bool:
    """Check if all expected columns are present."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing string values with empty string."""
    return df.fillna("")

def report_record_counts(df: pd.DataFrame, source_name: str) -> int:
    """Report the number of records."""
    count = len(df)
    print(f"{source_name}: {count} records")
    return count

def report_country_distribution(df: pd.DataFrame, source_name: str):
    """Report country distribution."""
    if 'country' in df.columns:
        print(f"Country distribution for {source_name}:")
        print(df['country'].value_counts(dropna=False).to_string())
    else:
        print(f"'country' column not found in {source_name}")

def load_ground_truth(filepath: str) -> pd.DataFrame:
    """Load and validate ground truth file."""
    gt_df = load_data(filepath)
    validate_columns(gt_df, ['source1_entity_id', 'matched_entity_ids'])
    gt_df = handle_missing_values(gt_df)
    return gt_df

def inspect_ground_truth(gt_df: pd.DataFrame):
    """Inspect ground truth match distributions."""
    total_s1 = len(gt_df)
    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if str(x).strip() else 0)
    
    print(f"Total Source 1 entities in GT: {total_s1}")
    print("Match count distribution:")
    print(gt_df['match_count'].value_counts().sort_index().to_string())
