import json
import pickle
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
import string

console = Console()

CHUNKS_PATH = Path("data/chunks/chunks.json")
BM25_INDEX_PATH = Path("data/processed/bm25_index.pkl")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "databricks_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many results each system retrieves before fusion
# We cast a wide net — top 20 from each
# RRF then combines and reranks to top 20 final
TOP_K = 20
RRF_CONSTANT = 60  # Standard constant, rarely needs changing


def tokenize(text: str) -> list[str]:
    """Must be identical to tokenizer in bm25_index.py."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


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

    # Process vector results
    for rank, chunk in enumerate(vector_results):
        key = get_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 1 + k)
        chunk_data[key] = chunk
        vector_seen.add(key)

    # Process BM25 results
    for rank, chunk in enumerate(bm25_results):
        key = get_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 1 + k)
        if key not in chunk_data:
            chunk_data[key] = chunk
        bm25_seen.add(key)

    # Sort by score
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


class HybridRetriever:
    """
    Main retrieval class that combines vector search and BM25.
    
    This is what the FastAPI endpoint will call directly.
    Encapsulating everything in a class means we load models
    once at startup, not on every query.
    """

    def __init__(self):
        console.print("[blue]Loading hybrid retriever...[/blue]")

        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Load ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        # Load BM25 index
        with open(BM25_INDEX_PATH, "rb") as f:
            index_data = pickle.load(f)
        self.bm25 = index_data["bm25"]
        self.chunks = index_data["chunks"]

        console.print("[green]✅ Hybrid retriever ready[/green]")

    def vector_search(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Semantic search using ChromaDB + MiniLM embeddings.
        Returns top k chunks by cosine similarity.
        """
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )

        # Reformat ChromaDB output into clean list of dicts
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
                "vector_score": 1 - distance  # Convert distance to similarity
            })

        return chunks

    def bm25_search(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Keyword search using BM25.
        Returns top k chunks by BM25 score.
        """
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]

        chunks = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            chunks.append(chunk)

        return chunks

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Main search method — runs both systems and fuses with RRF.
        This is the only method the rest of the pipeline calls.
        """
        # Run both searches independently
        vector_results = self.vector_search(query, k=k)
        bm25_results = self.bm25_search(query, k=k)

        # Fuse with RRF
        fused_results = reciprocal_rank_fusion(vector_results, bm25_results)

        return fused_results


def test_hybrid_search():
    retriever = HybridRetriever()

    queries = [
        "How do I create a Delta table?",
        "DELTA_TABLE_NOT_FOUND error fix",
        "MLflow experiment tracking setup"
    ]

    def get_key(chunk):
        return f"{chunk['source_url']}::{chunk['text'][:50]}"

    for query in queries:
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold]Query:[/bold] {query}")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")

        vector_results = retriever.vector_search(query, k=20)
        bm25_results = retriever.bm25_search(query, k=20)

        # Build lookup sets for display
        vector_keys = {get_key(r): i+1 for i, r in enumerate(vector_results)}
        bm25_keys = {get_key(r): i+1 for i, r in enumerate(bm25_results)}

        # Fuse
        hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results)

        console.print("\n[cyan]Vector Search Top 3:[/cyan]")
        for i, r in enumerate(vector_results[:3]):
            console.print(f"  {i+1}. [{r['vector_score']:.3f}] {r['source_url'].split('/')[-1]}")

        console.print("\n[magenta]BM25 Top 3:[/magenta]")
        for i, r in enumerate(bm25_results[:3]):
            console.print(f"  {i+1}. [{r['bm25_score']:.3f}] {r['source_url'].split('/')[-1]}")

        console.print("\n[green]Hybrid RRF Top 5:[/green]")
        for i, r in enumerate(hybrid_results[:5]):
            key = get_key(r)
            v_rank = vector_keys.get(key, "-")
            b_rank = bm25_keys.get(key, "-")
            in_both = v_rank != "-" and b_rank != "-"
            consensus = "[bold red]BOTH[/bold red]" if in_both else "one"
            console.print(
                f"  {i+1}. [{r['rrf_score']:.4f}] "
                f"V:{v_rank} B:{b_rank} "
                f"({consensus}) "
                f"{r['source_url'].split('/')[-1]}"
            )


# def debug_key_matching():
#     """
#     Check if chunk keys actually match between vector and BM25 systems.
#     This will tell us exactly why consensus isn't being detected.
#     """
#     retriever = HybridRetriever()
#     query = "How do I create a Delta table?"

#     vector_results = retriever.vector_search(query, k=20)
#     bm25_results = retriever.bm25_search(query, k=20)

#     def get_chunk_key(chunk):
#         return f"{chunk['source_url']}::{chunk['text'][:50]}"

#     vector_keys = {get_chunk_key(r): r['source_url'].split('/')[-1] 
#                    for r in vector_results}
#     bm25_keys = {get_chunk_key(r): r['source_url'].split('/')[-1] 
#                  for r in bm25_results}

#     # Find overlapping keys
#     common = set(vector_keys.keys()) & set(bm25_keys.keys())

#     console.print(f"\n[bold]Vector keys (first 3):[/bold]")
#     for k in list(vector_keys.keys())[:3]:
#         console.print(f"  '{k[:80]}'")

#     console.print(f"\n[bold]BM25 keys (first 3):[/bold]")
#     for k in list(bm25_keys.keys())[:3]:
#         console.print(f"  '{k[:80]}'")

#     console.print(f"\n[bold]Common keys:[/bold] {len(common)}")
#     for k in common:
#         console.print(f"  '{k[:80]}'")

if __name__ == "__main__":
    test_hybrid_search()