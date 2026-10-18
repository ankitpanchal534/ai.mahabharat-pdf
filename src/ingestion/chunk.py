import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

INPUT = Path("data/corpus/mahabharata_raw.jsonl")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def load_records():
    records = []

    with INPUT.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
            # chunks = splitter.split_text(text)

    return records

def create_section_document(section_records):
    # 1. Combine all record text
    # 2. Find first and last page
    # 3. Create and return a LangChain Document
    section_text = "\n\n".join(
    record["text"]
    for record in section_records
    )
    page_start = min(record["page"] for record in section_records)
    page_end = max(record["page"] for record in section_records)

    document = Document(
    page_content=section_text,
    metadata={
        "book": section_records[0]["book"],
        "section": section_records[0]["section"],
        "page_start": page_start,
        "page_end": page_end,
        "source": section_records[0]["source"],
    },
    )
    return document





if __name__ == "__main__":
    records = load_records()
    sections = set(
    record["section"]
    for record in records
    if record["section"] is not None
    )

    # print("Number of sections:", len(sections))
    # print(sorted(sections)[:20])
    sections = {}

    for record in records:
        if record["type"] not in {"paragraph", "list_item"}:
            continue

        section = record["section"]

        if section is None:
            continue
        if section not in sections:
            sections[section] = []

        sections[record["section"]].append(record)
    # print("Number of sections:", len(sections))
    # print("SECTION I records:", len(sections["SECTION I"]))
    # print(len(sections))
    # print(len(sections["SECTION I"]))
    # print(sections["SECTION I"][0])
    # print(sections["SECTION I"][-1])
    documents = []
    for section, section_records in sections.items():
        document = create_section_document(section_records)
        documents.append(document)
    # print("Number of documents:", len(documents))
    print(documents[0].metadata)
    chunks = splitter.split_documents(documents)
    print("Number of chunks:", len(chunks))
    print(chunks[0].metadata)
    print(chunks[0].page_content[:300])
    print("Length:", len(chunks[0].page_content))
    # document = create_section_document(sections["SECTION I"])
    
    # print(document.metadata)
    # print(document.page_content[:200])
    
  

    # chunks = splitter.split_documents([document])

    # for i, chunk in enumerate(chunks):
    #     print(f"\n--- Chunk {i} ---")
    #     print("Length:", len(chunk.page_content))
    #     print("Metadata:", chunk.metadata)
    #     print("Text:", chunk.page_content[:200])
    # print(document)
    # print(document.page_content[:200])
    # print(document.metadata)