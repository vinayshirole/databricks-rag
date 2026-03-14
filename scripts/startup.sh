#!/bin/bash
set -e

echo "🚀 Starting Databricks RAG service..."

# Step 1 — Build ChromaDB index if collection doesn't exist
echo "Checking vector index..."
python3 << 'PYEOF'
import sys
sys.path.append('.')
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
collections = client.list_collections()
names = [c.name for c in collections]
if 'databricks_docs' not in names:
    print('Building vector index...')
    from src.retrieval.vector_store import build_vector_index
    build_vector_index()
    print('Vector index built')
else:
    print('Vector index already exists')
PYEOF

# Step 2 — Start FastAPI
echo "🚀 Starting FastAPI..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000