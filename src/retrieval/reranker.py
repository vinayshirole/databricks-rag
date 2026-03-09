import torch
from sentence_transformers import CrossEncoder
from rich.console import Console

console = Console()

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 5


class Reranker:

    def __init__(self):
        console.print("[blue]Loading reranker model...[/blue]")
        self.model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        console.print("[green]✅ Reranker ready[/green]")

    def rerank(self, query: str, chunks: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
        if not chunks:
            return []
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
        return reranked[:top_k]