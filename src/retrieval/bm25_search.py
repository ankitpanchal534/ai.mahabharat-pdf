
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

INPUT = Path("data/corpus/mahabharata_chunks.jsonl")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    with INPUT.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    chunks = load_chunks()
    print("Chunks loaded:", len(chunks))

    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    question = "Krishna Arjuna duty Bhagavad Gita"
    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:5]

    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]

        print(f"\n--- Result {rank} ---")
        print("BM25 score:", round(float(scores[idx]), 4))
        print("Book:", chunk.get("book"))
        print("Section:", chunk.get("section"))
        print("Pages:", chunk.get("page_start"), "-", chunk.get("page_end"))
        print("Text:", chunk["text"][:500])
