
import hashlib
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


INPUT = Path("data/corpus/mahabharata_chunks.jsonl")
DB_PATH = "data/vectorstore/mahabharata"

TOP_K = 20
FINAL_K = 5
RRF_K = 60

# Resources will be initialized only once per Python process.
_resources = None


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def chunk_key(text: str, metadata: dict) -> str:
    """Create a consistent key for matching corpus and vector results."""
    content = text + json.dumps(
        metadata,
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_chunks() -> list[dict]:
    """Load the chunked Mahabharata corpus."""
    with INPUT.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def get_resources():
    """Initialize and cache BM25, chunks, lookup, and Chroma."""
    global _resources

    if _resources is not None:
        return _resources

    chunks = load_chunks()

    if not chunks:
        raise ValueError(f"No chunks found in {INPUT}")

    # Build BM25 keyword-search index.
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    # Load the existing persistent vector index.
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    vector_store = Chroma(
        collection_name="mahabharata",
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    # Map each corpus chunk to its stable key.
    chunk_lookup = {
        chunk_key(
            chunk["text"],
            {key: value for key, value in chunk.items()
             if key != "text"},
        ): chunk
        for chunk in chunks
    }

    _resources = (
        chunks,
        bm25,
        vector_store,
        chunk_lookup,
    )

    print(f"Hybrid search initialized: {len(chunks)} chunks loaded.")

    return _resources


def hybrid_search(
    question: str,
    top_k: int = TOP_K,
    final_k: int = FINAL_K,
) -> list[dict]:
    """
    Search using vector similarity + BM25 + Reciprocal Rank Fusion.

    Returns a list of dictionaries containing:
    text, metadata, and rrf_score.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1 or final_k < 1:
        raise ValueError("top_k and final_k must be positive.")

    chunks, bm25, vector_store, chunk_lookup = get_resources()

    # Accumulate Reciprocal Rank Fusion scores.
    rrf_scores = {}

    # 1. Vector search.
    vector_results = vector_store.similarity_search(
        question,
        k=top_k,
    )

    for rank, doc in enumerate(vector_results, start=1):
        key = chunk_key(doc.page_content, doc.metadata)

        rrf_scores[key] = (
            rrf_scores.get(key, 0.0)
            + 1 / (RRF_K + rank)
        )

    # 2. BM25 keyword search.
    bm25_scores = bm25.get_scores(tokenize(question))

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )[:top_k]

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        metadata = {
            key: value
            for key, value in chunk.items()
            if key != "text"
        }

        key = chunk_key(chunk["text"], metadata)

        rrf_scores[key] = (
            rrf_scores.get(key, 0.0)
            + 1 / (RRF_K + rank)
        )

    # 3. Combine rankings and select the final results.
    ranked_keys = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )[:final_k]

    results = []

    for key in ranked_keys:
        chunk = chunk_lookup.get(key)

        # Skip candidates that cannot be matched to the corpus.
        if chunk is None:
            continue

        metadata = {
            field: value
            for field, value in chunk.items()
            if field != "text"
        }

        results.append({
            "text": chunk["text"],
            "metadata": metadata,
            "rrf_score": rrf_scores[key],
        })

    return results


if __name__ == "__main__":
    question = (
        "In the Bhagavad Gita, what does Krishna teach Arjuna "
        "about doing his duty without attachment to results?"
    )

    results = hybrid_search(question)

    print(f"\nQuestion: {question}")
    print(f"Results returned: {len(results)}")

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\n--- Hybrid Result {rank} ---")
        print("RRF score:", round(result["rrf_score"], 5))
        print("Book:", metadata.get("book"))
        print("Section:", metadata.get("section"))
        print(
            "Pages:",
            metadata.get("page_start"),
            "-",
            metadata.get("page_end"),
        )
        print("Text:", result["text"][:500])
