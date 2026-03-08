import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

CHUNKS_PATH = Path("data/chunks/chunks.json")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "databricks_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

def build_vector_index():
    # Load chunks
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    console.print(f"[blue]Loaded {len(chunks)} chunks[/blue]")

    # Load embedding model
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    # Fast, lightweight, good quality for technical docs
    console.print("[blue]Loading embedding model...[/blue]")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Init ChromaDB with persistent storage
    # PersistentClient saves to disk so you don't rebuild every run
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if rebuilding from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
        console.print("[yellow]Deleted existing collection[/yellow]")
    except:
        pass

    # Create collection with cosine similarity
    # cosine is better than l2 (euclidean) for text embeddings
    # because it measures angle not magnitude
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 100
        }
    )

    # Embed and store in batches
    # Batching is critical — embedding one chunk at a time is 64x slower
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding chunks"):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{
            "source_url": c["source_url"],
            "title": c["title"],
            "token_count": c["token_count"],
            "has_code": str(c["has_code"])  # ChromaDB needs strings not bools
        } for c in batch]

        # This is where MiniLM converts text → 384-dim vectors
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True  # Required for cosine similarity
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    # Summary
    table = Table(title="Vector Index Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total vectors", str(collection.count()))
    table.add_row("Embedding model", EMBEDDING_MODEL)
    table.add_row("Embedding dimensions", "384")
    table.add_row("Similarity metric", "Cosine")
    table.add_row("HNSW M", "16")
    table.add_row("Storage path", CHROMA_PATH)
    console.print(table)

    return collection


# --- Quick test of the vector search to verify everything is working --- #
def test_vector_search(query: str = "How do I create a Delta table?"):
    """
    Quick sanity check — run one query and print top 5 results.
    If results look relevant, the index is working correctly.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # Embed the query the same way we embedded chunks
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    console.print(f"\n[bold]Test query:[/bold] '{query}'")
    console.print("[bold]Top 5 results:[/bold]\n")

    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        console.print(f"[cyan]Rank {i+1}[/cyan] | Score: {1-distance:.3f} | {meta['title']}")
        console.print(f"[blue]URL:[/blue] {meta['source_url']}")
        console.print(f"[white]{doc[:150]}...[/white]\n")


if __name__ == "__main__":
    build_vector_index()
    test_vector_search()