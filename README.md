# Databricks Documentation RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for answering technical questions over Databricks documentation. Built with hybrid search, cross-encoder reranking, citation verification, and hallucination detection.

---

## Architecture

```
User Query
    ↓
Hybrid Retrieval (Vector Search + BM25)
    ↓
Reciprocal Rank Fusion
    ↓
Cross-Encoder Reranking
    ↓
LLM Generation with Citations (Groq / Llama 3.1 8B)
    ↓
Citation Verification
    ↓
Verified Answer with Sources
```

---

## Key Features

- **Hybrid Search** — combines semantic vector search (MiniLM-L6-v2) with BM25 keyword search via Reciprocal Rank Fusion. Vector search alone scored 0.53 on technical queries; hybrid retrieval improved precision significantly.
- **Cross-Encoder Reranking** — ms-marco-MiniLM-L-6-v2 reads query and chunk jointly for precise relevance scoring, unlike bi-encoders which encode independently.
- **Citation Verification** — every LLM claim is verified against its cited source chunk. Hallucination rate tracked per response.
- **Semantic Chunking** — token-based chunking with code block preservation. Code blocks are never split across chunk boundaries.
- **Production API** — FastAPI with lifespan model loading, structured JSON responses, Prometheus-ready logging.
- **Dockerized** — 533MB optimized image with CPU-only PyTorch.

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB (HNSW, cosine similarity) |
| Keyword Search | BM25 (rank-bm25, k1=1.5, b=0.75) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.1 8B via Groq API |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Data | 483 Databricks documentation pages, 5,470 chunks |

---

## Project Structure

```
databricks-rag/
├── src/
│   ├── ingestion/
│   │   ├── scraper.py          # Sitemap-based doc scraper
│   │   └── data_inspector.py   # Data quality validation
│   ├── chunking/
│   │   └── chunker.py          # Token-based semantic chunker
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB HNSW index
│   │   ├── bm25_index.py       # BM25 inverted index
│   │   ├── hybrid_retriever.py # RRF fusion + deduplication
│   │   └── reranker.py         # Cross-encoder reranker
│   ├── generation/
│   │   └── generator.py        # Grounded generation + verification
│   └── api/
│       └── main.py             # FastAPI service
├── data/
│   ├── chunks/                 # 5,470 processed chunks
│   └── processed/              # BM25 index
├── scripts/
│   └── startup.sh              # Container startup script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker
- Groq API key (free at console.groq.com)

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/databricks-rag
cd databricks-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Build indexes (one time)
python3 src/ingestion/scraper.py
python3 src/chunking/chunker.py
python3 src/retrieval/vector_store.py
python3 src/retrieval/bm25_index.py

# Start the API
uvicorn src.api.main:app --reload --port 8000
```

### Docker

```bash
# Build and run
docker-compose up --build

# Test
curl http://localhost:8000/health
```

---

## API Reference

### `POST /query`

**Request:**
```json
{
  "query": "How do I create a Delta table?",
  "top_k_retrieval": 20,
  "top_k_rerank": 5
}
```

**Response:**
```json
{
  "query": "How do I create a Delta table?",
  "answer": "You can create a Delta table using several methods...[0]",
  "citations_used": [0, 1, 2],
  "chunks_used": 5,
  "hallucination_rate": 0.0,
  "verification": [
    {
      "claim": "You can use CREATE TABLE syntax...",
      "cited_chunk": 0,
      "cited_source": "https://docs.databricks.com/...",
      "match_ratio": 0.82,
      "verified": true,
      "reason": "8/10 key words found in chunk"
    }
  ],
  "latency_ms": 2835.26,
  "model": "llama-3.1-8b-instant"
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

---

## Design Decisions

**Why hybrid search over vector-only?**
Vector search alone scored 0.53 cosine similarity on technical queries — weak signal. BM25 covers exact keyword matches (error codes, API names, version numbers) that vector search misses. Hybrid RRF combines both and rewards consensus between systems.

**Why token-based chunking over recursive character splitting?**
Recursive splitting uses character count which is an unreliable proxy for token count in code-heavy documentation. Token-based chunking with explicit code block preservation gives predictable chunk sizes and prevents splitting SQL/Python examples at boundaries.

**Why a staged retrieval pipeline (retrieve → rerank)?**
Cross-encoders are O(n) — running joint encoding on 5,470 chunks per query would take minutes. Staged retrieval keeps P95 latency under 3,000ms: fast approximate retrieval narrows to 20 candidates, accurate reranking picks the final 5.

**Why Groq over local Ollama for production?**
Local Llama 3.2 3B on CPU: ~60,000ms per query. Groq API with Llama 3.1 8B: ~1,500ms per query. Better model, 40x faster, free tier sufficient for portfolio demo.

**Why citation verification?**
A RAG system without verification has no way to distinguish grounded answers from hallucinations. Citation verification catches the most dangerous pattern: fabricated facts with plausible-looking citations. Hallucination rate is tracked as a first-class metric.

---

## Evaluation

| Metric | Value |
|---|---|
| Corpus size | 483 pages, 5,470 chunks |
| Vocabulary (BM25) | 27,384 unique terms |
| Avg chunk size | 242 tokens |
| Code chunks | 1,018 (18.6%) |
| API latency (P95) | ~2,800ms (Groq) |
| Hallucination rate | 0-33% (query dependent) |
| Docker image size | 533MB |

### Known Limitations

- Corpus gap: some specific error codes (e.g. DELTA_TABLE_NOT_FOUND) not present in scraped 483 pages
- Lexical citation verifier produces false positives on paraphrased claims (~15%)
- Bullet point claims not verified by sentence splitter
- P95 latency increases to ~11,000ms on emulated linux/amd64 (Docker on M-series Mac)

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| GROQ_API_KEY | Yes | Groq API key from console.groq.com |
| QDRANT_URL | Optional | Qdrant Cloud cluster URL |
| QDRANT_API_KEY | Optional | Qdrant Cloud API key |
