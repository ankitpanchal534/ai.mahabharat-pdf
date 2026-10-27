
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DB_PATH = "data/vectorstore/mahabharata"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

vector_store = Chroma(
    collection_name="mahabharata",
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)

# question = "What does Krishna tell Arjuna about performing one's duty?"
question = "Krishna Arjuna duty Bhagavad Gita"

results = vector_store.similarity_search_with_score(
    question,
    k=5,
)

for i, (doc, score) in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Distance:", round(score, 4))
    print("Book:", doc.metadata.get("book"))
    print("Section:", doc.metadata.get("section"))
    print("Pages:", doc.metadata.get("page_start"), "-", doc.metadata.get("page_end"))
    print("Text:", doc.page_content[:500])
