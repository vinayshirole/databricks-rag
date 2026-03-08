import json
import uuid
import tiktoken
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# Configuration
CHUNK_SIZE = 512        # max tokens per chunk
CHUNK_OVERLAP = 50      # overlap tokens between chunks
MIN_CHUNK_SIZE = 50     # discard chunks smaller than this

INPUT_PATH = Path("data/raw/databricks_docs_raw.json")
OUTPUT_PATH = Path("data/chunks")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# cl100k_base is the tokenizer used by most modern models
# We use it to count tokens accurately, not characters
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def chunk_document(doc: dict) -> list[dict]:
    """
    Converts one scraped page into a list of chunks.
    
    Strategy:
    - Code blocks → always their own chunk, never split
    - Text sections → accumulate until CHUNK_SIZE, then flush
    - Last section of previous chunk → carried into next for overlap
    """
    chunks = []
    sections = doc["sections"]

    # Buffer that accumulates sections until we hit CHUNK_SIZE
    current_texts = []
    current_tokens = 0
    current_has_code = False

    def flush_chunk():
        """
        Takes whatever is in the buffer and saves it as a chunk.
        Called when buffer is full or document ends.
        """
        if not current_texts:
            return

        text = "\n\n".join(current_texts)

        # Discard tiny chunks — not enough context to be useful
        if count_tokens(text) < MIN_CHUNK_SIZE:
            return

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": text,
            "source_url": doc["url"],
            "title": doc["title"],
            "token_count": count_tokens(text),
            "has_code": current_has_code,
            "char_count": len(text)
        })

    for section in sections:
        text = section["text"]
        is_code = section["is_code"]
        token_count = count_tokens(text)

        # --- Rule 1: Code blocks are sacred ---
        # Never mix code with surrounding text
        # Never split a code block across chunks
        if is_code:
            # Save whatever text we had before this code block
            flush_chunk()
            current_texts.clear()
            current_tokens = 0
            current_has_code = False

            # Code block becomes its own standalone chunk
            if token_count >= MIN_CHUNK_SIZE:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": text,
                    "source_url": doc["url"],
                    "title": doc["title"],
                    "token_count": token_count,
                    "has_code": True,
                    "char_count": len(text)
                })
            continue

        # --- Rule 2: Flush when buffer would overflow ---
        if current_tokens + token_count > CHUNK_SIZE:
            flush_chunk()

            # OVERLAP: carry the last section into the next chunk
            # This is what makes chunks self-contained
            if current_texts:
                overlap_text = current_texts[-1]
                current_texts = [overlap_text]
                current_tokens = count_tokens(overlap_text)
            else:
                current_texts = []
                current_tokens = 0
            current_has_code = False

        # --- Rule 3: Handle sections longer than CHUNK_SIZE ---
        # Some paragraphs are just very long — split by sentences
        if token_count > CHUNK_SIZE:
            sentences = text.split(". ")
            temp_texts = []
            temp_tokens = 0

            for sentence in sentences:
                sent_tokens = count_tokens(sentence)

                if temp_tokens + sent_tokens > CHUNK_SIZE:
                    if temp_texts:
                        joined = ". ".join(temp_texts)
                        chunks.append({
                            "chunk_id": str(uuid.uuid4()),
                            "text": joined,
                            "source_url": doc["url"],
                            "title": doc["title"],
                            "token_count": count_tokens(joined),
                            "has_code": False,
                            "char_count": len(joined)
                        })
                    temp_texts = [sentence]
                    temp_tokens = sent_tokens
                else:
                    temp_texts.append(sentence)
                    temp_tokens += sent_tokens

            # Whatever's left goes into the main buffer
            if temp_texts:
                current_texts.extend(temp_texts)
                current_tokens += temp_tokens
            continue

        # --- Default: add section to buffer ---
        current_texts.append(text)
        current_tokens += token_count
        if is_code:
            current_has_code = True

    # Flush whatever remains at end of document
    flush_chunk()

    return chunks


def run_chunker() -> list[dict]:
    with open(INPUT_PATH) as f:
        docs = json.load(f)

    console.print(f"[blue]Chunking {len(docs)} documents...[/blue]")

    all_chunks = []

    for doc in tqdm(docs, desc="Chunking"):
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)

    # Save to disk
    output_file = OUTPUT_PATH / "chunks.json"
    with open(output_file, "w") as f:
        json.dump(all_chunks, f, indent=2)

    # Quality report
    token_counts = [c["token_count"] for c in all_chunks]
    code_chunks = sum(1 for c in all_chunks if c["has_code"])

    table = Table(title="Chunking Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total chunks", str(len(all_chunks)))
    table.add_row("Code chunks", str(code_chunks))
    table.add_row("Text chunks", str(len(all_chunks) - code_chunks))
    table.add_row("Avg tokens/chunk", f"{sum(token_counts)/len(token_counts):.0f}")
    table.add_row("Min tokens", str(min(token_counts)))
    table.add_row("Max tokens", str(max(token_counts)))

    console.print(table)
    console.print(f"\n[bold green]✅ Saved to {output_file}[/bold green]")

    return all_chunks


if __name__ == "__main__":
    run_chunker()