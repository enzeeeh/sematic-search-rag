import sys
import os
# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import DataLoader
import pandas as pd

def test_data_loading():
    """Test data loading with your Amazon dataset"""
    
    # Initialize data loader
    loader = DataLoader()
    
    # Since we're in tests folder, go up one level to find data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for Amazon dataset in data/raw folder
    possible_paths = [
        os.path.join(base_path, "data/raw/amazon.csv"),           # Your file
        # os.path.join(base_path, "data/raw/amazon_products.csv"),
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Found dataset: {path}")
            break
    
    if not dataset_path:
        print("❌ Amazon dataset not found!")
        print("📁 Looking in these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Base path: {base_path}")
        return None
    
    try:
        print(f"🚀 Testing data loader with: {dataset_path}")
        
        # Load dataset
        df = loader.load_amazon_dataset(dataset_path)
        
        print("\n" + "="*60)
        print("📊 DATASET SUMMARY")
        print("="*60)
        print(f"Total products: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        print("\n📋 SAMPLE DATA:")
        print("-"*60)
        print(df.head(3))
        
        print("\n📈 DATA TYPES:")
        print("-"*30)
        print(df.dtypes)
        
        print("\n📊 BRAND DISTRIBUTION (Top 10):")
        print("-"*40)
        print(df['brand'].value_counts().head(10))
        
        print("\n📊 CATEGORY DISTRIBUTION (Top 10):")
        print("-"*45)
        print(df['category'].value_counts().head(10))
        
        print("\n💰 PRICE STATISTICS:")
        print("-"*30)
        print(df['price'].describe())
        print(f"Currency range: ₹{df['price'].min():.2f} - ₹{df['price'].max():.2f}")
        
        print("\n📝 DESCRIPTION SAMPLE:")
        print("-"*35)
        sample_desc = df['description'].iloc[0]
        print(f"First product description (first 200 chars):")
        print(f"'{sample_desc[:200]}...'")
        
        # Validate some products
        print(f"\n✅ VALIDATION TEST:")
        print("-"*30)
        valid_products = loader.validate_products(df.head(100))
        print(f"Validated {len(valid_products)} out of 100 products successfully")
        
        # Create processed directory if it doesn't exist
        processed_dir = os.path.join(base_path, "data/processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save processed data
        output_path = os.path.join(processed_dir, "amazon_processed.csv")
        df.to_csv(output_path, index=False)
        print(f"\n💾 Processed data saved to: {output_path}")
        
        print("\n" + "="*60)
        print("🎉 DATA LOADING TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_data_loading()