import re
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase and strip extra whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_name(name: str) -> str:
    """Normalize business names by handling punctuation and common abbreviations."""
    name = clean_text(name)
    if not name:
        return name
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Normalize legal suffixes
    replacements = {
        r'\bcorp\b': 'corporation',
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated',
        r'\bco\b': 'company',
        r'\bllc\b': 'limited liability company'
    }
    
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)
        
    # Clean up multiple spaces left by punctuation removal
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_address(address: str) -> str:
    """Normalize addresses by expanding common abbreviations."""
    address = clean_text(address)
    if not address:
        return address
    
    # Remove punctuation (except maybe hyphens for zip codes, but let's keep it simple for now)
    address = re.sub(r'[^\w\s-]', ' ', address)
    
    replacements = {
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bsq\b': 'square',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment',
        r'\bpkwy\b': 'parkway',
        r'\bhwy\b': 'highway'
    }
    
    for pattern, replacement in replacements.items():
        address = re.sub(pattern, replacement, address)
        
    address = re.sub(r'\s+', ' ', address).strip()
    return address

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the business_name and business_address columns."""
    # Create a copy to avoid SettingWithCopyWarning
    processed_df = df.copy()
    
    if 'business_name' in processed_df.columns:
        processed_df['norm_name'] = processed_df['business_name'].apply(normalize_name)
    
    if 'business_address' in processed_df.columns:
        processed_df['norm_address'] = processed_df['business_address'].apply(normalize_address)
        
    if 'country' in processed_df.columns:
        # Keep country as is, but strip whitespace and lowercase for consistency
        processed_df['norm_country'] = processed_df['country'].astype(str).str.lower().str.strip()
        
    return processed_df
