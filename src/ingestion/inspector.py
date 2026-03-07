import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def inspect_scraped_data():
    path = Path("data/raw/databricks_docs_raw.json")
    
    with open(path) as f:
        docs = json.load(f)

    # Summary stats
    total_chars = sum(d["char_count"] for d in docs)
    avg_chars = total_chars / len(docs)
    has_code = sum(
        1 for d in docs
        if any(s["is_code"] for s in d["sections"])
    )

    table = Table(title="Scraped Data Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total pages", str(len(docs)))
    table.add_row("Total characters", f"{total_chars:,}")
    table.add_row("Avg chars/page", f"{avg_chars:.0f}")
    table.add_row("Pages with code blocks", str(has_code))
    table.add_row("Smallest page", f"{min(d['char_count'] for d in docs):,} chars")
    table.add_row("Largest page", f"{max(d['char_count'] for d in docs):,} chars")

    console.print(table)

    # Show 3 sample docs
    console.print("\n[bold]Sample docs:[/bold]")
    for doc in docs[:3]:
        console.print(f"\n[blue]URL:[/blue] {doc['url']}")
        console.print(f"[blue]Title:[/blue] {doc['title']}")
        console.print(f"[blue]Chars:[/blue] {doc['char_count']}")
        console.print(f"[blue]Preview:[/blue] {doc['raw_text'][:200]}...")

    # Find outlier pages  ← ADDED INSIDE THE FUNCTION
    console.print("\n[bold red]Outlier pages (>100K chars):[/bold red]")
    outliers = [d for d in docs if d["char_count"] > 100_000]
    console.print(f"Count: {len(outliers)}")
    for doc in outliers:
        console.print(f"\nURL: {doc['url']}")
        console.print(f"Chars: {doc['char_count']:,}")
        console.print(f"Title: {doc['title']}")
        console.print(f"Preview: {doc['raw_text'][:300]}")

if __name__ == "__main__":
    inspect_scraped_data()