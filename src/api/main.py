import time
import sys
sys.path.append(".")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import generate_answer

console = Console()


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to answer from Databricks docs"
    )
    top_k_retrieval: int = Field(default=20, ge=5, le=50)
    top_k_rerank: int = Field(default=5, ge=1, le=10)


class CitationResult(BaseModel):
    claim: str
    cited_chunk: int
    cited_source: str
    match_ratio: float
    verified: bool
    reason: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations_used: list[int]
    chunks_used: int
    hallucination_rate: float
    verification: list[CitationResult]
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str


# ─────────────────────────────────────────────
# Lifespan — Load models once at startup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when server starts.
    Loads all models into app.state so every request
    can reuse them without reloading from disk.

    This is the correct pattern for ML models in FastAPI.
    Loading inside the endpoint function would add 10+ seconds
    of latency to every single request.
    """
    console.print("[bold blue]Starting up — loading models...[/bold blue]")

    start = time.time()

    # Load hybrid retriever (embedding model + ChromaDB + BM25)
    app.state.retriever = HybridRetriever()

    # Load cross-encoder reranker
    app.state.reranker = Reranker()

    elapsed = time.time() - start
    console.print(f"[bold green]✅ Models loaded in {elapsed:.1f}s[/bold green]")
    console.print("[bold green]🚀 API ready to serve requests[/bold green]")

    yield  # Server runs here — handles all requests

    # Shutdown cleanup (if needed)
    console.print("[yellow]Shutting down...[/yellow]")


# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────

app = FastAPI(
    title="Databricks RAG API",
    description="Production RAG system for Databricks documentation Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allows frontend/browser to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """
    Health check endpoint.
    Used by deployment platforms to verify service is running.
    Always implement this — Railway/Render use it for monitoring.
    """
    return HealthResponse(
        status="healthy",
        models_loaded=hasattr(request.app.state, "retriever"),
        version="1.0.0"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Main RAG endpoint.

    Pipeline:
    1. Hybrid retrieval (vector + BM25 + RRF)
    2. Cross-encoder reranking
    3. LLM generation with citations
    4. Citation verification

    Returns structured response with answer, citations,
    verification results, and hallucination rate.
    """
    start = time.time()

    try:
        # Step 1 — Hybrid retrieval
        retriever = request.app.state.retriever
        hybrid_chunks = retriever.search(
            body.query,
            k=body.top_k_retrieval
        )

        if not hybrid_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found for this query"
            )

        # Step 2 — Rerank
        reranker = request.app.state.reranker
        final_chunks = reranker.rerank(
            body.query,
            hybrid_chunks,
            top_k=body.top_k_rerank
        )

        # Step 3 — Generate with citations
        result = generate_answer(body.query, final_chunks)

        # Step 4 — Compute latency
        latency_ms = (time.time() - start) * 1000

        # Log to console for monitoring
        console.print(
            f"[green]Query:[/green] {body.query[:50]}... | "
            f"[blue]Latency:[/blue] {latency_ms:.0f}ms | "
            f"[{'red' if result['hallucination_rate'] > 0.3 else 'green'}]"
            f"Hallucination: {result['hallucination_rate']:.0%}[/]"
        )

        # Build verification results
        verification = [
            CitationResult(**v)
            for v in result["verification"]
            if "cited_source" in v  # Skip out-of-range citations
        ]

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            citations_used=result["citations_used"],
            chunks_used=result["chunks_used"],
            hallucination_rate=result["hallucination_rate"],
            verification=verification,
            latency_ms=round(latency_ms, 2),
            model=result["model"]
        )

    except HTTPException:
        raise

    except Exception as e:
        console.print(f"[red]Error processing query: {e}[/red]")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "Databricks RAG API",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query"
    }