import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import DataLoader
from src.data_processing.text_processor import TextProcessor
from src.data_processing.embedding_generator import EmbeddingGenerator
import pandas as pd

def test_complete_pipeline():
    """Test the complete data processing pipeline"""
    
    print("🚀 Testing Complete Data Processing Pipeline")
    print("="*70)
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Load Amazon data
    print("\n📊 Step 1: Loading Amazon Dataset")
    print("-" * 40)
    
    loader = DataLoader()
    amazon_path = os.path.join(base_path, "data/raw/amazon.csv")
    
    if not os.path.exists(amazon_path):
        print("❌ Amazon dataset not found!")
        return None
    
    df = loader.load_amazon_dataset(amazon_path)
    sample_df = df.head(20)  # Use 20 products for complete test
    print(f"✅ Loaded {len(sample_df)} products for testing")
    
    # Step 2: Text Processing
    print("\n📝 Step 2: Text Processing & Chunking")
    print("-" * 40)
    
    text_processor = TextProcessor(target_chunk_size=125, overlap_size=25)
    chunks_df = text_processor.process_dataframe(sample_df)
    print(f"✅ Created {len(chunks_df)} chunks")
    
    # Step 3: Embedding Generation
    print("\n🤖 Step 3: Embedding Generation & Storage")
    print("-" * 40)
    
    embedding_generator = EmbeddingGenerator(
        persist_directory=os.path.join(base_path, "data/embeddings")
    )
    collection = embedding_generator.process_chunks_to_embeddings(chunks_df)
    
    # Step 4: Test Search Functionality
    print("\n🔍 Step 4: Testing Search Functionality")
    print("-" * 40)
    
    test_queries = [
        "charging cable for phone",
        "bluetooth headphones music",
        "laptop computer work"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = embedding_generator.test_similarity_search(collection, query, n_results=2)
    
    # Pipeline Summary
    print(f"\n📈 Pipeline Summary:")
    print("-" * 30)
    print(f"   Products processed: {len(sample_df)}")
    print(f"   Chunks created: {len(chunks_df)}")
    print(f"   Embeddings stored: {collection.count()}")
    print(f"   Embedding dimension: {embedding_generator.embedding_dim}")
    
    print("\n🎉 Complete pipeline test successful!")
    print("="*70)
    
    return {
        'dataframe': sample_df,
        'chunks': chunks_df,
        'collection': collection,
        'embedding_generator': embedding_generator
    }

if __name__ == "__main__":
    test_complete_pipeline()