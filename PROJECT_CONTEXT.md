Mahabharata Agentic AI — Project Handoff / Context File
Last updated: 2026-10-27

This file is a project handoff so another ChatGPT/Antigravity session can continue the work without relying on the previous conversation. It captures the project status and the next steps as known from the conversation. Treat uncertain or unverified details as things to check, not facts.
1. Project goal
Build a beginner-friendly Agentic AI application around an English-language Mahabharata PDF.

The immediate goal is a reliable, citation-first RAG system. Longer-term roadmap:
1. Phase 1 — citation-first RAG over the cleaned corpus
2. Phase 2 — knowledge graph / GraphRAG experiments
3. Phase 3 — agent/router layer (possibly LangGraph)
4. Phase 4 — evaluation and comparison
Do not jump to agents/GraphRAG before the baseline retrieval pipeline works. Learn and implement incrementally; explain concepts and give one small task at a time. The user prefers to write code themselves and get hints/reviews rather than receive a huge script.
2. User's working style and preferences
- Beginner to agentic AI / LangGraph; learning by building.
- Wants step-by-step coaching: explain the why, then give one manageable task.
- Prefers VS Code on Mac and uses uv.
- Wants practical commands and clear expected outputs.
- Conversational Hinglish (“bhidu” is welcome), but keep technical explanations clear.
- Avoid dumping full scripts unless asked.
- User asked to skip the recent Python cleanup part in this handoff; do not spend time on it unless asked.
3. Project location and important files
Project root:
/Users/ankitkumar/Downloads/python-ai-app

Source PDF:
complete_mahabharat.pdf

Likely PDF path:
data/raw/complete_mahabharat.pdf

Extraction JSON:
data/extracted/complete_mahabharat/complete_mahabharat.json

Normalized corpus JSONL:
data/corpus/mahabharata_raw.jsonl

Chunking code:
src/ingestion/chunk.py

Normalization code:
src/ingestion/normalize.py

Project environment:
- .venv exists and was recreated recently.
- uv is used for dependency/project management.
- pyproject.toml has requires-python = ">=3.13".
- uv.lock exists.
- The user verified:
  uv run python -c "import langchain, langgraph, langchain_text_splitters; print('RAG environment ready')"
  Output: RAG environment ready.
- Current project environment is working at least for those imports.
4. Pipeline and completed work
Conceptual pipeline:

PDF
→ OpenDataLoader extraction
→ raw extracted JSON
→ normalization into JSONL
→ group content by book + section
→ create LangChain Documents with provenance metadata
→ RecursiveCharacterTextSplitter
→ save chunks JSONL
→ embeddings
→ vector index
→ retrieval / reranking
→ LLM answer with citations
→ evaluation
Extraction
OpenDataLoader extracted the PDF. The PDF has 2,328 pages.

Extracted JSON path:
data/extracted/complete_mahabharat/complete_mahabharat.json

Top-level JSON fields previously observed:
- file name
- number of pages
- author
- title
- creation date
- modification date
- kids
number of pages: 2328
kids: 9998 elements

Element types:
- paragraph: 7773
- list: 2163
- heading: 62
Content extraction notes:
- 7,835 elements had direct content.
- 2,163 list elements had no direct content because their text was nested under list items.
- Lists/list items contain real prose and section labels; normalization must preserve nested list-item text.
- Headings include book titles and some section/end headings.
- There are 18 books.
Normalized corpus
Normalizer:
src/ingestion/normalize.py

Output:
data/corpus/mahabharata_raw.jsonl

Previously generated:
- Input elements: 9,998
- Output records: 14,199
- Record types:
  - heading: 62
  - paragraph: 7,773
  - section: 1,843
  - list_item: 4,521
Book counts previously observed:
- None: 2
- BOOK 1 ADI PARVA: 1485
- BOOK 2 SABHA PARVA: 430
- BOOK 3 VANA PARVA: 1445
- BOOK 4 VIRATA PARVA: 353
- BOOK 5 UDYOGA PARVA: 801
- BOOK 6 BHISHMA PARVA: 1054
- BOOK 7 DRONA PARVA: 1108
- BOOK 8 Karna-parva: 364
- BOOK 9 Shalya-parva: 289
- BOOK 10 Sauptika-parva: 163
- BOOK 11 Stri-parva: 78
- BOOK 12 SANTI PARVA: 3824
- BOOK 13 ANUSASANA PARVA: 1816
- BOOK 14 ASWAMEDHA PARVA: 656
- BOOK 15 ASRAMAVASIKA PARVA: 193
- BOOK 16 Mausala-parva: 40
- BOOK 17 Mahaprasthanika-parva: 32
- BOOK 18 Svargarohanika-parva: 66
The normalizer tracks:
- current_book from headings starting with BOOK
- current_section from list items starting with SECTION
Some records can have book=None or section=None, especially title/front matter and book-level material before a section.
5. Chunking implementation / current known status
File:
src/ingestion/chunk.py

It imports:
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
Input:
INPUT = Path("data/corpus/mahabharata_raw.jsonl")
Splitter:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
load_records() reads the JSONL file line by line and returns a list of dictionaries.

Function implemented:
def create_section_document(section_records):
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
The function was tested successfully for SECTION I:
- book: BOOK 1 ADI PARVA
- section: SECTION I
- page_start: 3
- page_end: 9
- source: complete_mahabharat.pdf
- first text begins: “Om! Having bowed down to Narayana and Nara, the most exalted male being...”
Previous grouping implementation used only section as key:
sections = {}
for record in records:
    if record["type"] not in {"paragraph", "list_item"}:
        continue
    section = record["section"]
    if section is None:
        continue
    if section not in sections:
        sections[section] = []
    sections[section].append(record)
This produced:
- 577 grouped section names
- SECTION I contained 44 records
The set of unique section labels from all records had previously shown 668 names, but that set included records of other types too. Do not assume the difference 668 vs 577 is an error by itself.
Last known chunking test
The code created a Document for each grouped section and called:
chunks = splitter.split_documents(documents)
Previously observed:
- 15,919 chunks
- first chunk metadata:
{
    "book": "BOOK 1 ADI PARVA",
    "section": "SECTION I",
    "page_start": 3,
    "page_end": 9,
    "source": "complete_mahabharat.pdf",
}
- first chunk text begins with the invocation “Om! Having bowed down to Narayana and Nara...”
- first chunk length: 138 characters
These 15,919 chunks were generated before correcting the grouping key and have NOT yet been saved to mahabharata_chunks.jsonl.
6. Critical next task: fix grouping by book + section
Before saving chunks, fix the grouping key to prevent sections with the same label in different books from being merged.

In src/ingestion/chunk.py, update grouping to use a tuple key:
sections = {}

for record in records:
    if record["type"] not in {"paragraph", "list_item"}:
        continue

    book = record["book"]
    section = record["section"]

    if section is None:
        continue

    key = (book, section)

    if key not in sections:
        sections[key] = []

    sections[key].append(record)

print("Book-section groups:", len(sections))
This is the next coding task. User has not yet run this corrected grouping or provided its count.

Potential improvement after this: decide how to handle records with book=None, and verify group contents are chronologically contiguous. Avoid silently discarding useful front matter if citations/content require it. For initial section-based RAG, it may be reasonable to filter out records with missing section, but note that it excludes front matter/book-level text.

After grouping:
1. Build documents = []
2. Loop for key, section_records in sections.items():
3. document = create_section_document(section_records)
4. append to documents
5. chunks = splitter.split_documents(documents)
6. Validate chunk sizes, metadata, and counts.
7. Save to data/corpus/mahabharata_chunks.jsonl, one JSON object per line. A good record schema:
{
  "chunk_id": "stable ID",
  "text": "...",
  "metadata": {
    "book": "...",
    "section": "...",
    "page_start": 3,
    "page_end": 9,
    "source": "complete_mahabharat.pdf"
  }
}
Make sure the serialization handles all chunk metadata values. Include a deterministic chunk ID (e.g., based on index initially; a content hash can be considered later). Do not overengineer before a working baseline.
7. RAG strategy agreed with user
User asked if RAG is dead. Agreed:
- RAG is not dead; naive vector-only RAG is often insufficient.
- Start with a baseline hybrid retrieval approach, then measure.
- Planned comparison:
  1. Vector search alone
  2. BM25 alone
  3. Hybrid retrieval
  4. Hybrid retrieval + reranker
- Evaluate with ~20–30 question/expected-evidence pairs.
- Add Knowledge Graph / GraphRAG only if it measurably helps multi-hop character/event relationship questions.
- Add agentic routing only after baseline retrieval works.
- Citation-first design: metadata comes from corpus, not invented by LLM.
- Preserve page/source provenance.
- Avoid blindly embedding corrupted or incorrectly grouped text.
8. Relevant dependencies (from pyproject.toml)
[project]
name = "python-ai-app"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "langchain>=1.4.3",
    "langchain-core>=1.6.6",
    "langchain-ollama>=1.1.0",
    "langchain-text-splitters>=1.1.2",
    "langgraph>=1.2.12",
    "ocrmac>=1.0.1",
    "onnxruntime>=1.30.0",
    "opendataloader-pdf[hybrid]>=2.5.11",
    "python-dotenv>=1.2.3",
    "rapidocr>=3.9.2",
]
uv.lock exists.
9. What not to do
- Do not redo Python cleanup unless asked; user explicitly said to skip that part for this handoff.
- Do not jump to LangGraph/agents before baseline RAG works.
- Do not use only vector embeddings without a retrieval evaluation plan.
- Do not mix section records from different books.
- Do not provide huge scripts without need. Continue with one small step at a time.
- Do not assume the 15,919 chunks are the final corrected chunk count.
- Do not claim chunks JSONL exists yet; it has not been saved in this workflow.
10. Suggested first response in the new session
Start naturally, acknowledge the handoff file and say something like:
“Bhai, I’ve got the context. We’re continuing from the chunking validation stage, not restarting. Open src/ingestion/chunk.py; next we’ll fix the grouping key from section to (book, section), run it, then persist the corrected chunks to JSONL.”

Then give only the next step and ask the user to send the output.