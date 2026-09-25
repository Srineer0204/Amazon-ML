import unittest
import pandas as pd
from src.preprocess import normalize_name, normalize_address, preprocess_dataframe

class TestPreprocess(unittest.TestCase):
    
    def test_normalize_name(self):
        self.assertEqual(normalize_name("Amazon Corp."), "amazon corporation")
        self.assertEqual(normalize_name("TATA MOTORS PVT LTD"), "tata motors private limited")
        self.assertEqual(normalize_name("  Google  Inc  "), "google incorporated")
        self.assertEqual(normalize_name("A & B Co."), "a b company")
        
    def test_normalize_address(self):
        self.assertEqual(normalize_address("123 Main St."), "123 main street")
        self.assertEqual(normalize_address("456 Park Ave, Ste 100"), "456 park avenue suite 100")
        self.assertEqual(normalize_address("Near SBI ATM, M.G. Rd"), "near sbi atm m g road")
        
    def test_preprocess_dataframe(self):
        data = {
            'entity_id': ['1', '2'],
            'business_name': ['Amazon Corp', 'Google Inc'],
            'business_address': ['123 Main St', '456 Park Ave'],
            'country': ['US', 'India']
        }
        df = pd.DataFrame(data)
        processed_df = preprocess_dataframe(df)
        
        self.assertIn('norm_name', processed_df.columns)
        self.assertIn('norm_address', processed_df.columns)
        self.assertIn('norm_country', processed_df.columns)
        
        self.assertEqual(processed_df['norm_name'].iloc[0], 'amazon corporation')
        self.assertEqual(processed_df['norm_address'].iloc[0], '123 main street')
        self.assertEqual(processed_df['norm_country'].iloc[0], 'us')

if __name__ == '__main__':
    unittest.main()
