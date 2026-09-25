import os
import unittest
import pandas as pd
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data import load_data, validate_columns, handle_missing_values, inspect_ground_truth

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Create dummy dataframe
        self.df = pd.DataFrame({
            'entity_id': ['S1-001', 'S1-002'],
            'business_name': ['A', None],
            'country': ['US', 'India']
        })
    
    def test_validate_columns(self):
        self.assertTrue(validate_columns(self.df, ['entity_id', 'country']))
        with self.assertRaises(ValueError):
            validate_columns(self.df, ['entity_id', 'missing_col'])
            
    def test_handle_missing_values(self):
        df_clean = handle_missing_values(self.df)
        self.assertEqual(df_clean['business_name'][1], "")
        
if __name__ == '__main__':
    unittest.main()
