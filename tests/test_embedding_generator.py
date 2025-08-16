import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.embedding_generator import EmbeddingGenerator
import pandas as pd

def test_embedding_generation():
    """Test embedding generation and ChromaDB storage"""
    
    print("🚀 Testing Embedding Generation & Vector Storage")
    print("="*60)
    
    # Load chunks data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_path = os.path.join(base_path, "data/processed/product_chunks.csv")
    
    if not os.path.exists(chunks_path):
        print("❌ No chunks data found. Run test_text_processor.py first!")
        return None
    
    print(f"📂 Loading chunks from: {chunks_path}")
    chunks_df = pd.read_csv(chunks_path)
    
    # Use subset for testing (first 50 chunks)
    test_chunks = chunks_df.head(50).copy()
    print(f"📊 Testing with {len(test_chunks)} chunks")
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory=os.path.join(base_path, "data/embeddings")
    )
    
    # Process chunks to embeddings
    collection = embedding_generator.process_chunks_to_embeddings(test_chunks)
    
    # Test similarity searches
    test_queries = [
        "iPhone cable charger",
        "wireless headphones",
        "laptop computer",
        "kitchen appliances",
        "smart watch"
    ]
    
    print(f"\n🔍 Testing similarity search with sample queries:")
    print("="*60)
    
    for query in test_queries:
        embedding_generator.test_similarity_search(collection, query, n_results=3)
        print()
    
    # Collection stats
    print(f"📊 ChromaDB Collection Statistics:")
    print(f"   Total chunks stored: {collection.count()}")
    print(f"   Embedding dimension: {embedding_generator.embedding_dim}")
    print(f"   Storage location: {embedding_generator.persist_directory}")
    
    print("\n🎉 Embedding generation test completed!")
    return collection

if __name__ == "__main__":
    test_embedding_generation()