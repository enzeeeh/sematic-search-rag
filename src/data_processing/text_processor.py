import nltk
from typing import List, Dict, Tuple
import re
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class TextProcessor:
    def __init__(self, target_chunk_size: int = 125, overlap_size: int = 25):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = 50
    
    def concatenate_product_text(self, product_row: Dict) -> str:
        """Concatenate title, brand, category, and description for better embeddings"""
        
        title = str(product_row.get('title', '')).strip()
        brand = str(product_row.get('brand', 'unknown')).strip()
        category = str(product_row.get('category', '')).strip()
        description = str(product_row.get('description', '')).strip()
        price = product_row.get('price', 0)
        
        # Create structured text with context
        concatenated_parts = []
        
        # Title repetition for importance
        if title:
            concatenated_parts.append(f"Product: {title}")
        
        # Brand and category context
        if brand and brand.lower() != 'unknown':
            concatenated_parts.append(f"Brand: {brand}")
        
        if category:
            # Clean category (remove special characters)
            clean_category = category.replace('&', ' and ').replace('|', ' > ')
            concatenated_parts.append(f"Category: {clean_category}")
        
        # Price context
        if price > 0:
            concatenated_parts.append(f"Price: ₹{price}")
        
        # Main description
        if description:
            concatenated_parts.append(f"Description: {description}")
        
        # Join all parts
        concatenated_text = ". ".join(concatenated_parts)
        
        return concatenated_text
    
    def chunk_text(self, text: str, product_id: str) -> List[Dict]:
        """Split text into optimal chunks with overlap"""
        
        if not text or len(text.strip()) < self.min_chunk_size:
            return [{
                'text': text,
                'chunk_id': f"{product_id}_chunk_0",
                'chunk_index': 0,
                'word_count': len(text.split()) if text else 0,
                'product_id': product_id
            }]
        
        # Clean text
        text = self._clean_text(text)
        
        # Sentence tokenization
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback if NLTK fails
            sentences = self._simple_sentence_split(text)
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence exceeds target size
            if current_word_count + sentence_words > self.target_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': f"{product_id}_chunk_{chunk_index}",
                    'chunk_index': chunk_index,
                    'word_count': current_word_count,
                    'product_id': product_id
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_word_count = len(current_chunk.split())
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_word_count += sentence_words
        
        # Add final chunk if it meets minimum size
        if current_chunk and current_word_count >= 30:  # Relaxed minimum for final chunk
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': f"{product_id}_chunk_{chunk_index}",
                'chunk_index': chunk_index,
                'word_count': current_word_count,
                'product_id': product_id
            })
        
        # If no valid chunks, create one with the original text
        if not chunks:
            chunks.append({
                'text': text,
                'chunk_id': f"{product_id}_chunk_0",
                'chunk_index': 0,
                'word_count': len(text.split()),
                'product_id': product_id
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better processing"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common formatting issues
        text = text.replace('|', '. ')
        text = text.replace('\n', '. ')
        text = text.replace('\t', ' ')
        
        # Remove extra periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting if NLTK fails"""
        # Split on periods, exclamation marks, question marks
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, text: str) -> str:
        """Get last N words for overlap between chunks"""
        if not text:
            return ""
        
        words = text.split()
        if len(words) > self.overlap_size:
            return " ".join(words[-self.overlap_size:])
        return ""
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe to create chunks"""
        print("🔄 Processing products for text chunking...")
        
        all_chunks = []
        
        for idx, row in df.iterrows():
            # Concatenate text
            concatenated_text = self.concatenate_product_text(row.to_dict())
            
            # Create chunks
            chunks = self.chunk_text(concatenated_text, row['product_id'])
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk_data = {
                    **chunk,  # chunk info
                    'original_title': row['title'],
                    'brand': row['brand'],
                    'category': row['category'],
                    'price': row['price'],
                    'availability': row['availability']
                }
                all_chunks.append(chunk_data)
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(df)} products")
        
        # Convert to DataFrame
        chunks_df = pd.DataFrame(all_chunks)
        
        # Add statistics
        word_counts = chunks_df['word_count']
        print(f"📊 Chunk statistics:")
        print(f"   Average words per chunk: {word_counts.mean():.1f}")
        print(f"   Min words: {word_counts.min()}")
        print(f"   Max words: {word_counts.max()}")
        print(f"   Target range (100-150): {((word_counts >= 100) & (word_counts <= 150)).sum()} chunks")
        
        return chunks_df