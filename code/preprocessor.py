# Save this as preprocessor.py
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        
    def extract_features(self, df):
        """Extract meaningful features from product data"""
        
        # Make a copy
        df_processed = df.copy()
        
        # 1. Extract year from title
        df_processed['product_year'] = df_processed['title'].apply(self._extract_year)
        
        # 2. Extract brand
        df_processed['brand'] = df_processed['title'].apply(self._extract_brand)
        
        # 3. Condition encoding
        df_processed['condition_encoded'] = self._encode_condition(df_processed['condition'])
        
        # 4. Age calculation
        current_year = datetime.now().year
        df_processed['age'] = current_year - df_processed['product_year']
        df_processed['age'] = df_processed['age'].apply(lambda x: max(x, 0))
        
        # 5. Title length as proxy for description quality
        df_processed['title_length'] = df_processed['title'].apply(lambda x: len(str(x)))
        
        # 6. Has defects (keywords in title)
        defect_keywords = ['cracked', 'broken', 'damaged', 'scratched', 'not working', 'damage']
        df_processed['has_defects'] = df_processed['title'].apply(
            lambda x: 1 if any(keyword in str(x).lower() for keyword in defect_keywords) else 0
        )
        
        # 7. Product category
        df_processed['category'] = df_processed['title'].apply(self._categorize_product)
        
        # Remove rows with missing price
        df_processed = df_processed[df_processed['price'] > 0]
        
        return df_processed
    
    def _extract_year(self, title):
        """Extract product year from title"""
        try:
            matches = re.findall(r'\b(19[0-9]{2}|20[0-1][0-9]|202[0-4])\b', str(title))
            if matches:
                return int(matches[0])
        except:
            pass
        return datetime.now().year - 3  # Default: 3 years old
    
    def _extract_brand(self, title):
        """Extract brand name from title"""
        brands = ['apple', 'samsung', 'sony', 'dell', 'hp', 'lenovo', 'nike', 'adidas', 'lg', 'google', 'microsoft']
        title_lower = str(title).lower()
        for brand in brands:
            if brand in title_lower:
                return brand
        return 'other'
    
    def _encode_condition(self, condition_series):
        """Encode condition text to numeric values"""
        condition_mapping = {
            'new': 5,
            'like new': 4,
            'excellent': 4,
            'very good': 3,
            'good': 3,
            'acceptable': 2,
            'used': 2,
            'fair': 1,
            'poor': 0
        }
        
        encoded = []
        for cond in condition_series:
            cond_lower = str(cond).lower()
            score = 2  # default for 'used'
            for key, value in condition_mapping.items():
                if key in cond_lower:
                    score = value
                    break
            encoded.append(score)
        
        return encoded
    
    def _categorize_product(self, title):
        """Categorize product based on keywords"""
        categories = {
            'electronics': ['iphone', 'samsung', 'laptop', 'tablet', 'camera', 'tv', 'macbook', 'ipad', 'phone'],
            'clothing': ['shirt', 'pants', 'jacket', 'shoe', 'dress', 'sneaker', 'jacket'],
            'furniture': ['chair', 'table', 'sofa', 'bed', 'desk'],
            'books': ['book', 'novel', 'textbook'],
            'automotive': ['car', 'tire', 'part', 'accessory']
        }
        
        title_lower = str(title).lower()
        for category, keywords in categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        return 'other'
    
    def encode_categorical(self, df, columns):
        """Encode categorical columns"""
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        return df