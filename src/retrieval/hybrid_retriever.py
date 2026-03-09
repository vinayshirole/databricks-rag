import json
import pickle
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rich.console import Console
import string

console = Console()

CHUNKS_PATH = Path("data/chunks/chunks.json")
BM25_INDEX_PATH = Path("data/processed/bm25_index.pkl")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "databricks_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 20
RRF_CONSTANT = 60


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if len(t) > 2]


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_CONSTANT
) -> list[dict]:
    rrf_scores = {}
    chunk_data = {}
    vector_seen = set()
    bm25_seen = set()

    def get_key(chunk: dict) -> str:
        return f"{chunk['source_url']}::{chunk['text'][:50]}"

    for rank, chunk in enumerate(vector_results):
        key = get_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 1 + k)
        chunk_data[key] = chunk
        vector_seen.add(key)

    for rank, chunk in enumerate(bm25_results):
        key = get_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 1 + k)
        if key not in chunk_data:
            chunk_data[key] = chunk
        bm25_seen.add(key)

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for key in sorted_keys[:TOP_K]:
        chunk = chunk_data[key].copy()
        chunk["rrf_score"] = rrf_scores[key]
        chunk["in_vector"] = key in vector_seen
        chunk["in_bm25"] = key in bm25_seen
        chunk["in_both"] = key in vector_seen and key in bm25_seen
        results.append(chunk)

    return results


def deduplicate_by_url(chunks: list[dict], max_per_url: int = 2) -> list[dict]:
    url_counts = {}
    deduplicated = []
    for chunk in chunks:
        url = chunk["source_url"]
        count = url_counts.get(url, 0)
        if count < max_per_url:
            deduplicated.append(chunk)
            url_counts[url] = count + 1
    return deduplicated


class HybridRetriever:

    def __init__(self):
        console.print("[blue]Loading hybrid retriever...[/blue]")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        with open(BM25_INDEX_PATH, "rb") as f:
            index_data = pickle.load(f)
        self.bm25 = index_data["bm25"]
        self.chunks = index_data["chunks"]
        console.print("[green]✅ Hybrid retriever ready[/green]")

    def vector_search(self, query: str, k: int = TOP_K) -> list[dict]:
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        chunks = []
        for doc, meta, distance, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0]
        ):
            chunks.append({
                "chunk_id": chunk_id,
                "text": doc,
                "source_url": meta["source_url"],
                "title": meta["title"],
                "vector_score": 1 - distance
            })
        return chunks

    def bm25_search(self, query: str, k: int = TOP_K) -> list[dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        chunks = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            chunks.append(chunk)
        return chunks

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        vector_results = self.vector_search(query, k=k)
        bm25_results = self.bm25_search(query, k=k)
        fused_results = reciprocal_rank_fusion(vector_results, bm25_results)
        return deduplicate_by_url(fused_results, max_per_url=2)