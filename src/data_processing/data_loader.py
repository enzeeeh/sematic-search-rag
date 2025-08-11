import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel, validator
import re
import numpy as np
from pathlib import Path

class ProductSchema(BaseModel):
    product_id: str
    title: str
    description: str
    brand: Optional[str] = "unknown"
    category: str
    price: float
    availability: bool = True
    
    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError('Title must be at least 5 characters')
        if len(v) > 200:
            v = v[:200]  # Truncate instead of error
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v:
            return "No description available"
        if len(v) > 1000:
            v = v[:1000]  # Truncate long descriptions
        return v.strip()
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return round(float(v), 2)
    
    @validator('brand')
    def validate_brand(cls, v):
        if not v or pd.isna(v):
            return "unknown"
        return str(v).lower().strip()

class DataLoader:
    def __init__(self):
        self.category_mapping = {
            'computers&accessories': 'electronics/computers',
            'electronics': 'electronics/general',
            'home&kitchen': 'home/kitchen',
            'toys&games': 'toys/games',
            'clothing&accessories': 'fashion/clothing',
            'sports&outdoors': 'sports/outdoor',
            'books': 'books/general',
            'beauty&personalcare': 'beauty/personal-care',
            'automotive': 'automotive/general',
            'health&wellness': 'health/wellness'
        }
    
    def load_amazon_dataset(self, filepath: str) -> pd.DataFrame:
        """Load and process Amazon product dataset"""
        print(f"📂 Loading dataset from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"📊 Columns: {list(df.columns)}")
            
            # Process the dataframe
            df_processed = self.process_amazon_data(df)
            print(f"✅ Dataset processed: {len(df_processed)} valid rows")
            
            return df_processed
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def process_amazon_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize Amazon dataset"""
        
        print(f"📋 Dataset columns: {list(df.columns)}")
        
        # Map your specific columns to our schema
        column_mapping = {
            'product_name': 'title',
            'about_product': 'description',
            'discounted_price': 'price',
            'category': 'category'
            # product_id already exists with correct name
        }
        
        print(f"📋 Using column mapping: {column_mapping}")
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                print(f"   ✅ Renamed {old_col} → {new_col}")
        
        print(f"📋 After renaming: {list(df.columns)}")
        
        # Clean and normalize the data
        return self.normalize_dataframe(df)
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the dataframe"""
        print("🧹 Normalizing data...")
        
        # Generate product IDs if not present
        if 'product_id' not in df.columns:
            df['product_id'] = 'prod_' + df.index.astype(str)
        
        # Handle missing values
        df['title'] = df['title'].fillna('Unknown Product').astype(str)
        df['description'] = df['description'].fillna('No description available').astype(str)
        
        # Extract/normalize brand
        df['brand'] = self.extract_brand(df)
        
        # Normalize category
        df['category'] = df.apply(self.normalize_category, axis=1)
        
        # Extract/normalize price
        df['price'] = self.extract_price(df)
        
        # Set availability
        df['availability'] = True
        
        # Clean text fields
        df['title'] = df['title'].str.strip()
        df['description'] = df['description'].str.strip()
        
        # Remove rows with invalid data
        initial_count = len(df)
        df = df[df['title'].str.len() >= 5]  # Minimum title length
        df = df[df['price'] > 0]  # Valid price
        df = df.dropna(subset=['title', 'description'])
        
        final_count = len(df)
        print(f"📊 Filtered: {initial_count} → {final_count} rows ({initial_count - final_count} removed)")
        
        # Select final columns
        final_columns = ['product_id', 'title', 'description', 'brand', 'category', 'price', 'availability']
        df = df[final_columns].copy()
        
        return df
    
    def extract_brand(self, df: pd.DataFrame) -> pd.Series:
        """Extract brand from available columns"""
        if 'brand' in df.columns:
            return df['brand'].fillna('unknown').astype(str).str.lower().str.strip()
        
        # Try to extract from title
        brands = []
        for title in df['title']:
            if pd.isna(title):
                brands.append('unknown')
                continue
            
            # Extract potential brand (first word, common patterns)
            words = str(title).split()
            if words:
                brands.append(words[0].lower().strip())
            else:
                brands.append('unknown')
        
        return pd.Series(brands)
    
    def normalize_category(self, row) -> str:
        """Normalize category to hierarchical format"""
        # Try different category column names
        category_cols = ['category', 'main_category', 'product_category', 'subcategory']
        
        category = None
        for col in category_cols:
            if col in row.index and pd.notna(row[col]):
                category = str(row[col]).strip().lower()
                break
        
        if not category:
            return 'general/uncategorized'
        
        # Apply category mapping
        for key, value in self.category_mapping.items():
            if key in category:
                return value
        
        # Create hierarchical format from category
        parts = re.split(r'[|,&\-\s]+', category)
        parts = [part.strip() for part in parts if part.strip()][:3]  # Max 3 levels
        
        return '/'.join(parts) if parts else 'general/uncategorized'
    
    def extract_price(self, df: pd.DataFrame) -> pd.Series:
        """Extract price from available columns"""
        price_cols = ['price', 'discounted_price', 'actual_price', 'selling_price']
        
        for col in price_cols:
            if col in df.columns:
                prices = df[col].copy()
                
                # Clean price strings (remove currency symbols, commas)
                if prices.dtype == 'object':
                    prices = prices.astype(str).str.replace(r'[₹$,£€]', '', regex=True)
                    prices = prices.str.replace(',', '')
                    prices = pd.to_numeric(prices, errors='coerce')
                
                # Use this column if it has valid prices
                valid_prices = prices[prices > 0]
                if len(valid_prices) > 0:
                    return prices.fillna(valid_prices.median())
        
        # If no price column found, generate random prices
        print("⚠️ No price column found, generating sample prices")
        np.random.seed(42)
        return pd.Series(np.random.uniform(10, 1000, len(df)))
    
    def validate_products(self, df: pd.DataFrame) -> List[ProductSchema]:
        """Validate products using Pydantic schema"""
        print("✅ Validating products...")
        
        valid_products = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                product = ProductSchema(**row.to_dict())
                valid_products.append(product)
            except Exception as e:
                errors.append(f"Row {idx}: {e}")
        
        print(f"✅ Validation complete: {len(valid_products)} valid, {len(errors)} errors")
        if errors[:5]:  # Show first 5 errors
            print("⚠️ Sample errors:", errors[:5])
        
        return valid_products