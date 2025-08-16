from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path
import time

class EmbeddingGenerator:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "./data/embeddings"):
        
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"📐 Embedding dimension: {self.embedding_dim}")
        
        # Initialize ChromaDB
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        print(f"🗄️ Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for text chunks in batches"""
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            batch = texts[i:i + batch_size]
            
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return np.array(embeddings)
    
    def create_collection(self, collection_name: str = "product_embeddings") -> chromadb.Collection:
        """Create or get ChromaDB collection"""
        
        # Delete existing collection if exists (for fresh start)
        try:
            existing_collection = self.client.get_collection(collection_name)
            self.client.delete_collection(collection_name)
            print(f"🗑️ Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection with cosine similarity
        collection = self.client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16
            }
        )
        
        print(f"✅ Created collection: {collection_name}")
        return collection
    
    def store_embeddings(self, 
                        chunks_df: pd.DataFrame, 
                        embeddings: np.ndarray,
                        collection_name: str = "product_embeddings") -> chromadb.Collection:
        """Store embeddings and metadata in ChromaDB"""
        
        print(f"💾 Storing {len(embeddings)} embeddings in ChromaDB...")
        
        # Create collection
        collection = self.create_collection(collection_name)
        
        # Prepare data for ChromaDB
        ids = chunks_df['chunk_id'].tolist()
        documents = chunks_df['text'].tolist()
        
        # Prepare metadata (ChromaDB requires string values)
        metadatas = []
        for idx, row in chunks_df.iterrows():
            metadata = {
                'product_id': str(row['product_id']),
                'chunk_index': str(row['chunk_index']),
                'word_count': str(row['word_count']),
                'brand': str(row['brand']),
                'category': str(row['category']),
                'price': str(row['price']),
                'availability': str(row['availability']),
                'title': str(row['original_title'][:100])  # Truncate for storage
            }
            metadatas.append(metadata)
        
        # Store in batches
        batch_size = 100
        total_batches = (len(ids) + batch_size - 1) // batch_size
        
        for i in range(0, len(ids), batch_size):
            batch_num = (i // batch_size) + 1
            end_idx = min(i + batch_size, len(ids))
            
            print(f"   Storing batch {batch_num}/{total_batches}")
            
            collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"✅ Stored all embeddings in collection: {collection_name}")
        return collection
    
    def process_chunks_to_embeddings(self, chunks_df: pd.DataFrame) -> chromadb.Collection:
        """Complete pipeline: chunks -> embeddings -> storage"""
        
        print(f"🚀 Starting embedding pipeline for {len(chunks_df)} chunks")
        print("="*60)
        
        # Extract texts
        texts = chunks_df['text'].tolist()
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self.generate_embeddings(texts)
        embedding_time = time.time() - start_time
        
        print(f"⏱️ Embedding generation took: {embedding_time:.2f} seconds")
        print(f"📊 Average time per text: {embedding_time/len(texts)*1000:.2f} ms")
        
        # Store in ChromaDB
        start_time = time.time()
        collection = self.store_embeddings(chunks_df, embeddings)
        storage_time = time.time() - start_time
        
        print(f"⏱️ Storage took: {storage_time:.2f} seconds")
        
        # Verify storage
        stored_count = collection.count()
        print(f"✅ Verification: {stored_count} items stored in ChromaDB")
        
        return collection
    
    def test_similarity_search(self, collection: chromadb.Collection, query: str, n_results: int = 5):
        """Test similarity search with a sample query"""
        
        print(f"\n🔍 Testing similarity search with query: '{query}'")
        print("-" * 50)
        
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search similar chunks
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"📋 Top {n_results} similar chunks:")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            print(f"\n{i+1}. Similarity: {similarity:.3f}")
            print(f"   Product: {metadata['title']}")
            print(f"   Brand: {metadata['brand']}")
            print(f"   Price: ₹{metadata['price']}")
            print(f"   Text: {doc[:100]}...")
        
        return results