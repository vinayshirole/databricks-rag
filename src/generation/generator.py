import re
import os
from groq import Groq
from rich.console import Console

console = Console()

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024

# Uses GROQ_API_KEY environment variable automatically
groq_client = Groq()


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the grounded prompt with citations.

    Key design decisions:
    1. Number each chunk with [0], [1] etc — LLM uses these as citation markers
    2. Explicit instruction to ONLY use provided context — grounding
    3. Explicit instruction to say I don't know — honesty
    4. Explicit instruction to cite every claim — verifiability
    5. Warn against adding outside knowledge — hallucination prevention
    """
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[{i}] Source: {chunk['title']} ({chunk['source_url']})\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a precise technical assistant for Databricks documentation.

    STRICT RULES YOU MUST FOLLOW:
    1. Answer ONLY using the context provided below. Do not use any outside knowledge.
    2. Cite every factual claim with the chunk number like [0], [1], [2] etc.
    3. If the context does not contain enough information to answer, say exactly: "I don't have enough information in the provided context to answer this question."
    4. Never invent facts, dates, version numbers, or details not present in the context.
    5. If you cite a chunk, the claim must be directly supported by that chunk's text.
    6. Do not write summary sentences that cite multiple chunks at once.
    7. Each sentence should cite exactly one chunk maximum.
    8. After your answer, list the source URLs for every chunk you cited.

    CONTEXT:
    {context}

    QUESTION: {query}

    ANSWER (cite every claim with [chunk_number], then list sources at the end):"""

    return prompt


def extract_citations(text: str) -> list[int]:
    """
    Extract all citation numbers from generated text.
    Finds patterns like [0], [1], [2] etc.
    """
    citations = re.findall(r'\[(\d+)\]', text)
    return [int(c) for c in citations]


def verify_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Verify each cited claim against its source chunk.

    For each sentence containing a citation:
    1. Extract the claim
    2. Find the cited chunk
    3. Check if key words from the claim appear in the chunk
    4. Flag as verified or hallucinated

    Lexical verification — catches fabricated facts with real citations.
    Known limitation: paraphrased claims score lower than exact matches.
    """
    verified_claims = []

    sentences = re.split(r'(?<=[.!?])\s+', answer)

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "to", "of", "in", "for", "on", "with", "by",
        "you", "can", "and", "or", "it", "this", "that"
    }
    
    skip_phrases = [
        "therefore", "in summary", "in conclusion",
        "to summarize", "thus", "overall", "so,",
        "sources:", "source:", "references:", "cited sources:"
    ]

    negative_phrases = [
        "not specified", "not mentioned", "does not",
        "doesn't", "not provided", "not covered",
        "not explain", "not contain"
    ]

    for sentence in sentences:
        if any(sentence.lower().startswith(p) for p in skip_phrases):
            continue
        if any(p in sentence.lower() for p in negative_phrases):
            continue

        citations = re.findall(r'\[(\d+)\]', sentence)

        if not citations:
            continue

        for cite_num in citations:
            cite_idx = int(cite_num)

            if cite_idx >= len(chunks):
                verified_claims.append({
                    "claim": sentence,
                    "cited_chunk": cite_idx,
                    "verified": False,
                    "reason": "Citation index out of range"
                })
                continue

            # Split camelCase and method names for code awareness
            raw_text = chunks[cite_idx]["text"].lower()
            chunk_text = re.sub(r'[._()"\']', ' ', raw_text)

            claim_clean = re.sub(r'\[\d+\]', '', sentence).lower()
            claim_words = [
                w.strip('.,!?;:')
                for w in claim_clean.split()
                if w.strip('.,!?;:') not in stop_words
                and len(w.strip('.,!?;:')) > 3
            ]

            matches = sum(1 for w in claim_words if w in chunk_text)
            match_ratio = matches / len(claim_words) if claim_words else 0

            # Lower threshold for code chunks —
            # natural language claims about code always score lower lexically
            is_code_chunk = chunks[cite_idx].get("has_code", "False") == "True"
            threshold = 0.25 if is_code_chunk else 0.4
            verified = match_ratio >= threshold

            verified_claims.append({
                "claim": sentence.strip(),
                "cited_chunk": cite_idx,
                "cited_source": chunks[cite_idx]["source_url"],
                "match_ratio": round(match_ratio, 2),
                "verified": verified,
                "reason": f"{matches}/{len(claim_words)} key words found in chunk"
            })

    return verified_claims


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Full generation pipeline:
    1. Build grounded prompt
    2. Generate with Llama 3.2
    3. Extract citations
    4. Verify citations
    5. Return structured response
    """
    prompt = build_prompt(query, chunks)

    console.print("[blue]Generating answer...[/blue]")
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        stop=["QUESTION:", "CONTEXT:"]
    )

    answer = response.choices[0].message.content.strip()

    citations_used = extract_citations(answer)
    verification_results = verify_citations(answer, chunks)

    if verification_results:
        hallucinated = sum(1 for v in verification_results if not v["verified"])
        hallucination_rate = hallucinated / len(verification_results)
    else:
        hallucination_rate = 0.0

    return {
        "query": query,
        "answer": answer,
        "citations_used": citations_used,
        "chunks_used": len(chunks),
        "verification": verification_results,
        "hallucination_rate": hallucination_rate,
        "model": GROQ_MODEL
    }


def test_generator():
    import sys
    sys.path.append(".")
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker

    retriever = HybridRetriever()
    reranker = Reranker()

    queries = [
        "How do I create a Delta table?",
        "What is MLflow experiment tracking?",
        "How do I configure autoscaling for a Databricks cluster?"
    ]

    for query in queries:
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold]Query:[/bold] {query}")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")

        hybrid_chunks = retriever.search(query, k=20)
        final_chunks = reranker.rerank(query, hybrid_chunks, top_k=5)
        result = generate_answer(query, final_chunks)

        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(result["answer"])

        console.print(f"\n[bold]Citation Verification:[/bold]")
        console.print(
            f"Hallucination rate: "
            f"[{'red' if result['hallucination_rate'] > 0 else 'green'}]"
            f"{result['hallucination_rate']:.0%}[/]"
        )

        for v in result["verification"]:
            status = "✅" if v["verified"] else "❌"
            console.print(
                f"\n{status} Claim: {v['claim'][:80]}..."
                f"\n   Cited chunk [{v['cited_chunk']}] | "
                f"Match: {v['match_ratio']:.0%} | "
                f"{v['reason']}"
            )


if __name__ == "__main__":
    test_generator()