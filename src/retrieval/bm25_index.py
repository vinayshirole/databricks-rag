import json
import pickle
import string
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

CHUNKS_PATH = Path("data/chunks/chunks.json")
BM25_INDEX_PATH = Path("data/processed/bm25_index.pkl")
BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

def tokenize(text: str) -> list[str]:
    """
    Convert text into a list of lowercase tokens.
    
    This is critical — BM25 does exact word matching.
    If query has "delta" but chunk has "Delta", they won't match
    unless we lowercase everything consistently.
    
    We also remove punctuation so "table." and "table" are the same word.
    """
    # Lowercase everything
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Split by whitespace
    tokens = text.split()
    
    # Remove very short tokens (1-2 chars) — usually noise
    tokens = [t for t in tokens if len(t) > 2]
    
    return tokens


def build_bm25_index():
    # Load chunks
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    
    console.print(f"[blue]Loaded {len(chunks)} chunks[/blue]")
    console.print("[blue]Tokenizing chunks...[/blue]")
    
    # Tokenize every chunk
    # This builds the inverted index and computes
    # term frequencies and chunk lengths upfront
    tokenized_chunks = []
    for chunk in tqdm(chunks, desc="Tokenizing"):
        tokens = tokenize(chunk["text"])
        tokenized_chunks.append(tokens)
    
    console.print("[blue]Building BM25 index...[/blue]")
    
    # BM25Okapi is the standard BM25 variant
    # k1=1.5 controls term frequency saturation
    # b=0.75 controls length normalization
    # These are industry standard defaults
    bm25 = BM25Okapi(
        tokenized_chunks,
        k1=1.5,
        b=0.75
    )
    
    # Save index + chunks together
    # We need chunks alongside bm25 to retrieve 
    # the actual text and metadata at query time
    index_data = {
        "bm25": bm25,
        "chunks": chunks
    }
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    
    # Stats
    avg_tokens = sum(len(t) for t in tokenized_chunks) / len(tokenized_chunks)
    vocab_size = len(bm25.idf)
    
    table = Table(title="BM25 Index Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total chunks indexed", str(len(chunks)))
    table.add_row("Vocabulary size", f"{vocab_size:,}")
    table.add_row("Avg tokens per chunk", f"{avg_tokens:.0f}")
    table.add_row("k1 (TF saturation)", "1.5")
    table.add_row("b (length normalization)", "0.75")
    table.add_row("Saved to", str(BM25_INDEX_PATH))
    console.print(table)
    
    return bm25, chunks


def test_bm25_search(query: str = "How do I create a Delta table?"):
    """
    Load saved index and run a test query.
    Compare these results with vector search results —
    you'll see BM25 finds different but complementary chunks.
    """
    with open(BM25_INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    
    bm25 = index_data["bm25"]
    chunks = index_data["chunks"]
    
    # Tokenize query the same way we tokenized chunks
    # Consistency is critical — same tokenizer for both
    query_tokens = tokenize(query)
    console.print(f"\n[blue]Query tokens:[/blue] {query_tokens}")
    
    # Get BM25 scores for all 5,470 chunks
    scores = bm25.get_scores(query_tokens)
    
    # Get top 5 indices sorted by score
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:5]
    
    console.print(f"\n[bold]BM25 results for:[/bold] '{query}'\n")
    
    for rank, idx in enumerate(top_indices):
        chunk = chunks[idx]
        score = scores[idx]
        console.print(f"[cyan]Rank {rank+1}[/cyan] | Score: {score:.3f} | {chunk['title']}")
        console.print(f"[blue]URL:[/blue] {chunk['source_url']}")
        console.print(f"[white]{chunk['text'][:150]}...[/white]\n")


if __name__ == "__main__":
    build_bm25_index()
    test_bm25_search()