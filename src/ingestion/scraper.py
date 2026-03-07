import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from urllib.parse import urljoin
from tqdm import tqdm
from rich.console import Console

console = Console()

BASE_URL = "https://docs.databricks.com"
SITEMAP_URL = "https://docs.databricks.com/sitemap.xml"
OUTPUT_PATH = Path("data/raw")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research bot; contact: your@email.com)"
}

EXCLUDE_PATTERNS = [
    "/archive/",          # Deprecated runtime release notes
    "/release-notes/",    # Changelogs, not how-to docs
    "runtime-release",    # Runtime version history
    "/system-tables/",    # Raw schema dumps
    "/audit-logs",        # Exhaustive event reference
    "function-reference", # Dashboard function dump
    "jobs-update",        # Migration scripts
    "flexible-node-type", # Instance type compatibility tables
]

# Sitemap is used to discover all doc URLs, then we scrape each page for content.
def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    console.print("[bold blue]Fetching sitemap...[/bold blue]")
    resp = requests.get(sitemap_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.content, "lxml-xml")
    urls = [loc.text for loc in soup.find_all("loc")]
    
    # Filter English docs only, skip redirects/assets
    filtered = [
        u for u in urls
        if "/en/" in u
        and not u.endswith((".png", ".jpg", ".pdf"))                        # Skip media files
        and not any(pattern in u for pattern in EXCLUDE_PATTERNS)           # Skip very long/irrelevant pages
    ]
    
    console.print(f"[green]Found {len(filtered)} URLs[/green]")
    return filtered


# Scrape a single page, extracting title and main content sections.
def scrape_page(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove noise
        for tag in soup.find_all(["nav", "footer", "aside", 
                                   "script", "style", "header"]):
            tag.decompose()

        # Get main content
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", {"class": "content"}) or
            soup.find("div", {"role": "main"})
        )
        if not main:
            return None

        # Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

        # Extract sections preserving structure
        sections = []
        for elem in main.find_all(["h1","h2","h3","h4","p","pre","li","code"]):
            text = elem.get_text(strip=True)
            if not text or len(text) < 10:
                continue
            sections.append({
                "type": elem.name,
                "text": text,
                "is_code": elem.name in ["pre", "code"]
            })

        if not sections:
            return None

        raw_text = "\n\n".join([s["text"] for s in sections])

        # Skip near-empty pages
        if len(raw_text) < 200:
            return None
        
        # Skip extremely long pages that are likely dumps or reference tables``
        if len(raw_text) > 100_000:
            console.print(f"[yellow]Skipping oversized page ({len(raw_text):,} chars): {url}[/yellow]")
            return None

        # Save everything as json for later processing
        return {
            "url": url,
            "title": title,
            "sections": sections,
            "raw_text": raw_text,
            "char_count": len(raw_text),
            "scraped_at": time.time()
        }

    except Exception as e:
        console.print(f"[red]Failed {url}: {e}[/red]")
        return None

def run_scraper(limit: int = 500) -> list[dict]:
    urls = get_urls_from_sitemap(SITEMAP_URL)
    urls = urls[:limit]

    results = []
    failed = 0

    for url in tqdm(urls, desc="Scraping docs"):
        doc = scrape_page(url)
        if doc:
            results.append(doc)
        else:
            failed += 1
        time.sleep(0.5)  # Time delay

    # Save raw
    output_file = OUTPUT_PATH / "databricks_docs_raw.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[bold green]✅ Scraped: {len(results)} pages[/bold green]")
    console.print(f"[yellow]Failed/Skipped: {failed}[/yellow]")
    console.print(f"[blue]Saved to: {output_file}[/blue]")

    return results

if __name__ == "__main__":
    run_scraper(limit=500)