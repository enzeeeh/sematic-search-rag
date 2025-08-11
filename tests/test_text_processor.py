import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.text_processor import TextProcessor
from src.data_processing.data_loader import DataLoader
import pandas as pd

def test_text_processing():
    """Test text processing and chunking"""
    
    print("🚀 Testing Text Processing & Chunking")
    print("="*50)
    
    # Initialize components
    text_processor = TextProcessor(target_chunk_size=125, overlap_size=25)
    
    # Load processed data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_path, "data/processed/amazon_processed.csv")
    
    if not os.path.exists(processed_path):
        print("❌ No processed data found. Run test_data_loader.py first!")
        return None
    
    print(f"📂 Loading processed data from: {processed_path}")
    df = pd.read_csv(processed_path)
    
    # Test with first 10 products for demo
    sample_df = df.head(10).copy()
    
    print(f"📊 Processing {len(sample_df)} products...")
    
    # Process text
    chunks_df = text_processor.process_dataframe(sample_df)
    
    print(f"\n📋 Sample concatenated text:")
    print("-" * 40)
    sample_product = sample_df.iloc[0].to_dict()
    concatenated = text_processor.concatenate_product_text(sample_product)
    print(f"Original title: {sample_product['title'][:50]}...")
    print(f"Concatenated text: {concatenated[:200]}...")
    
    print(f"\n📄 Sample chunks:")
    print("-" * 30)
    sample_chunks = chunks_df[chunks_df['product_id'] == sample_df.iloc[0]['product_id']]
    
    for idx, chunk in sample_chunks.iterrows():
        print(f"Chunk {chunk['chunk_index']}: {chunk['word_count']} words")
        print(f"Text: {chunk['text'][:100]}...")
        print()
    
    # Save chunks
    chunks_output_path = os.path.join(base_path, "data/processed/product_chunks.csv")
    chunks_df.to_csv(chunks_output_path, index=False)
    print(f"💾 Chunks saved to: {chunks_output_path}")
    
    print("\n🎉 Text processing test completed!")
    return chunks_df

if __name__ == "__main__":
    test_text_processing()