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
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if len(t) > 2]


def build_bm25_index():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    console.print(f"[blue]Loaded {len(chunks)} chunks[/blue]")

    tokenized_chunks = []
    for chunk in tqdm(chunks, desc="Tokenizing"):
        tokens = tokenize(chunk["text"])
        tokenized_chunks.append(tokens)

    bm25 = BM25Okapi(tokenized_chunks, k1=1.5, b=0.75)

    index_data = {"bm25": bm25, "chunks": chunks}

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)

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


if __name__ == "__main__":
    build_bm25_index()