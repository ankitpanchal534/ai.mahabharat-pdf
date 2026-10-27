
import json
import re
import hashlib
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

INPUT = Path("data/corpus/mahabharata_chunks.jsonl")
DB_PATH = "data/vectorstore/mahabharata"

TOP_K = 5
CANDIDATES = 20

QUESTIONS = [
    "What does Krishna teach Arjuna about work without attachment?",
    "Why does Krishna say action is better than inaction?",
    "What does Krishna say about performing one's own duty?",
]

EXPECTED_BOOK = "BOOK 6 BHISHMA PARVA"
EXPECTED_SECTION = "SECTION XXVII"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def chunk_key(text, metadata):
    content = text + json.dumps(
        metadata, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_chunks():
    with INPUT.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_key(chunk):
    metadata = {k: v for k, v in chunk.items() if k != "text"}
    return chunk_key(chunk["text"], metadata)


if __name__ == "__main__":
    chunks = load_chunks()
    print("Chunks loaded:", len(chunks))

    bm25 = BM25Okapi([
        tokenize(chunk["text"]) for chunk in chunks
    ])

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    vector_store = Chroma(
        collection_name="mahabharata",
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    for question in QUESTIONS:
        rrf_scores = {}

        # Vector candidates
        vector_results = vector_store.similarity_search(
            question, k=CANDIDATES
        )

        for rank, doc in enumerate(vector_results, start=1):
            key = chunk_key(doc.page_content, doc.metadata)
            rrf_scores[key] = (
                rrf_scores.get(key, 0) + 1 / (60 + rank)
            )

        # BM25 candidates
        scores = bm25.get_scores(tokenize(question))
        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:CANDIDATES]

        for rank, idx in enumerate(indices, start=1):
            key = get_key(chunks[idx])
            rrf_scores[key] = (
                rrf_scores.get(key, 0) + 1 / (60 + rank)
            )

        # Combined ranking
        ranked_keys = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:TOP_K]

        relevant_ranks = []

        for rank, key in enumerate(ranked_keys, start=1):
            chunk = next(
                (c for c in chunks if get_key(c) == key),
                None,
            )

            if chunk is None:
                continue

            if (
                chunk.get("book") == EXPECTED_BOOK
                and chunk.get("section") == EXPECTED_SECTION
            ):
                relevant_ranks.append(rank)

        print(f"\nQuestion: {question}")

        if relevant_ranks:
            print("Target section found at rank:", relevant_ranks)
        else:
            print("Target section found in top 5: NO")

        print("Top 5 sections:")
        for rank, key in enumerate(ranked_keys, start=1):
            chunk = next(
                (c for c in chunks if get_key(c) == key),
                None,
            )
            if chunk:
                print(
                    rank,
                    chunk.get("book"),
                    "|",
                    chunk.get("section"),
                )
