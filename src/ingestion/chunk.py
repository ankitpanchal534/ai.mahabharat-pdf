import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

INPUT = Path("data/corpus/mahabharata_raw.jsonl")
OUTPUT = Path("data/corpus/mahabharata_chunks.jsonl")

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
    return records

def create_section_document(section_records):
    # count = 0

    # for record in records:
    #     if (
    #         record["type"] in {"paragraph", "list_item"}
    #         and record["section"] == "SECTION I"
    #         and record["book"] == "BOOK 1 ADI PARVA"
    #     ):
    #         print("Page:", record["page"])
    #         print("Text:", record["text"][:100])
    #         print("---")

    #         count += 1
    #         if count == 5:
    #             break
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
    all_sections = {}
    # print("record 18 ",records[18])
    for record in records:
        if record["type"] not in {"paragraph", "list_item"}:
            continue

        book = record["book"]
        section = record["section"]

        if section is None:
           continue
        key = (book, section)

        if key not in all_sections:
            all_sections[key] = []

        all_sections[key].append(record)
    documents = []
    for section, section_records in all_sections.items():
        document = create_section_document(section_records)
        documents.append(document)
    chunks = splitter.split_documents(documents)
    # chunk = chunks[18]
    # print("Length:", len(chunk.page_content))
    # print("Metadata:", chunk.metadata)
    # print("Text:", chunk.page_content)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "text": chunk.page_content,
                **chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {OUTPUT}")
  
