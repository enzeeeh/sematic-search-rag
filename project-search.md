    # Product Catalog Semantic Search (RAG) Documentation

## System Overview
An end-to-end retrieval-augmented generation (RAG) pipeline for product search that understands natural language queries and returns JSON-formatted results with confidence scoring.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffd8d8'}, 'config': {'fontSize': 10, 'flowchart': {'rankSpacing': 15, 'nodeSpacing': 10, 'wrap': true, 'curve': 'basis'}}}%%
flowchart TD
    %% ========== TECHNICAL NOTES ==========
    noteChunk["**Chunk Size (100-150 words)**<br>• Matches BERT's 128-token context<br>• Preserves product feature relationships<br>• Avoids over-fragmentation"]:::note
    noteMeta["**Metadata Schema**<br>• category: hierarchical (electronics/audio)<br>• brand: normalized lowercase<br>• price_band: discrete ranges (0-50, 50-100)"]:::note
    noteJSON["**JSON Prompt Format**<br>• Strict JSON-only output<br>• Includes chunk citations<br>• Example: <pre>{'product_id':'123', 'citation':'chunk2'}</pre>"]:::note
    noteConf["**Confidence Formula**<br>0.7 × (cosine similarity)<br>+ 0.3 × (metadata match %)<br>Threshold: 0.6"]:::note
    noteFallback["**Fallback Strategy**<br>1. Hybrid search (BM25+embeddings)<br>2. Relax filters progressively<br>3. Return top result with warning"]:::note
    noteHybrid["**Hybrid Search**<br>α=0.7 (embeddings)<br>+ 0.3 (BM25 keyword score)<br>Boosts rare term recall"]:::note
    notePerf["**Performance Optimizations**<br>• Pre-filter by metadata<br>• HNSW index for ANN<br>• Cache frequent queries"]:::note

    %% ========== MAIN FLOW ==========
    subgraph DataProcessing["Data Preparation"]
        direction TB    
        C[("fa:fa-database Raw Data")]:::blue
        D["Normalize Fields"]:::blue
        E["Validate Schema"]:::blue
        F["Title+Desc Concatenate"]:::blue
        G["Chunk Text (100-150 words)"]:::blue
        C --> D --> E --> F --> G
    end
    
    subgraph EmbeddingStorage["Embedding Generation"]
        direction TB
        H["Generate Embeddings"]:::purple
        I[("fa:fa-cube Vector DB")]:::purple
        G --> H
        H --> I
    end
    
    subgraph QueryProcessing["Query Execution"]
        direction TB
        B[/"Query Processing"/]:::orange
        J["Extract Filters"]:::orange
        K["Metadata Filtering"]:::orange
        L["Query Embedding"]:::orange
        M["Semantic Search"]:::orange
        B --> J --> K --> L --> M
    end
    
    subgraph ResponseGen["Response Generation"]
        direction TB
        N["Combine Context"]:::pink
        O["JSON Prompt + Citations"]:::pink
        P["Confidence Score"]:::pink
        Q{"Confidence > 0.6?"}:::yellow
        R[("fa:fa-file-code JSON Response")]:::green
        S["Fallback: Hybrid Search"]:::red
        T["Relax Filters"]:::red
        M ==> N
        N --> O --> P --> Q
        Q -- Yes --> R
        Q -- No --> S --> T -.-> P
    end

    %% ========== ORTHOGONAL CONNECTIONS ==========
    A[("fa:fa-user User Query")]:::green ==> B
    I -->|"Pre-filtered<br>by metadata"| K
    R --> U[("fa:fa-user User")]:::green
    
    %% ========== STYLING ==========
    classDef green fill:#B2DFDB,stroke:#00897B,stroke-width:1px
    classDef orange fill:#FFE0B2,stroke:#FB8C00,stroke-width:1px
    classDef blue fill:#BBDEFB,stroke:#1976D2,stroke-width:1px
    classDef yellow fill:#FFF9C4,stroke:#FBC02D,stroke-width:1px,shape:diamond
    classDef pink fill:#F8BBD0,stroke:#C2185B,stroke-width:1px
    classDef purple fill:#E1BEE7,stroke:#8E24AA,stroke-width:1px
    classDef red fill:#FFCDD2,stroke:#E53935,stroke-width:1px
    classDef note fill:#FFFDE7,stroke:#FBC02D,stroke-width:1px,text-align:left
    
    style DataProcessing stroke:#1976D2,stroke-dasharray:5 2,fill:#E3F2FD20
    style EmbeddingStorage stroke:#8E24AA,stroke-dasharray:5 2,fill:#F3E5F520
    style QueryProcessing stroke:#FB8C00,stroke-dasharray:5 2,fill:#FFF3E020
    style ResponseGen stroke:#C2185B,stroke-dasharray:5 2,fill:#FCE4EC20

    %% ========== NOTE ATTACHMENTS ==========
    noteChunk -.- G
    noteMeta -.- I
    noteJSON -.- O
    noteConf -.- P
    noteFallback -.- S
    noteHybrid -.- M
    notePerf -.- K
```

## Key Components

### 1. Data Processing
- **Input**: Raw product data (title, description, brand, price)
- **Steps**:
  - Normalization (lowercase, special chars removal)
  - Concatenation (title + description)
  - Chunking (100-150 words per chunk)
- **Why 100-150 words?**  
  Optimal balance for embedding models (preserves context without fragmentation)

### 2. Metadata Schema
```text
{
  "text": "Product description...",
  "metadata": {
    "category": "electronics/audio",  // hierarchical
    "brand": "sony",                 // lowercase
    "price_band": "100-200"          // predefined ranges
  }
}
```

### 3. Query Processing
| Step | Description |
|------|-------------|
| Filter Extraction | Pulls brand/category/price from natural language |
| Metadata Filter | First-stage filtering before vector search |
| Hybrid Search | BM25 + embeddings (α=0.7/0.3) |

### 4. Response Format
```json
{
  "results": [{
    "product_id": "123",
    "confidence": 0.82,
    "citation": "chunk2_product123"
  }]
}
```

## Confidence Scoring
`score = (0.7 × semantic_similarity) + (0.3 × metadata_match)`  
**Threshold**: 0.6 (below triggers fallback)

## Fallback Strategy
1. Hybrid search (BM25 + embeddings)
2. Progressive filter relaxation
3. Final fallback: Top result with low-confidence flag

## Performance Optimizations
- **Pre-filtering**: Reduces search space by 60%
- **HNSW Index**: Approximate nearest neighbor search
- **Caching**: Frequent query results (5min TTL)

*Total document size: ~450 words (fits one A4 page at 11pt font)*