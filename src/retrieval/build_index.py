
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

INPUT = Path("data/corpus/mahabharata_chunks.jsonl")
DB_PATH = "data/vectorstore/mahabharata"

def load_chunks():
    documents = []

    with INPUT.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            text = record.pop("text")

            documents.append(
                Document(
                    page_content=text,
                    metadata=record,
                )
            )

    return documents


if __name__ == "__main__":
    documents = load_chunks()
    print("Chunks loaded:", len(documents))

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    vector_store = Chroma(
        collection_name="mahabharata",
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    batch_size = 100

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        ids = [
            f"chunk-{i}"
            for i in range(start, start + len(batch))
        ]
        vector_store.add_documents(
            documents=batch,
            ids=ids,
        )
        print(f"Indexed {min(start + len(batch), len(documents))}/{len(documents)}")

    print("Vector index ready:", DB_PATH)
