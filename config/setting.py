import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 125
CHUNK_OVERLAP = 25

# Search configuration
CONFIDENCE_THRESHOLD = 0.6
TOP_K_RESULTS = 10
SEMANTIC_WEIGHT = 0.7
METADATA_WEIGHT = 0.3

# Performance settings
BATCH_SIZE = 32
MAX_WORKERS = 4

# Amazon dataset configuration
AMAZON_DATASET_PATH = RAW_DATA_DIR / "amazon_products.csv"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)