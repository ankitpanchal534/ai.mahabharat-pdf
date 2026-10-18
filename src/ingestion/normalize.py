import json
from pathlib import Path


INPUT = Path(
    "data/extracted/complete_mahabharat/complete_mahabharat.json"
)

OUTPUT = Path(
    "data/corpus/mahabharata_raw.jsonl"
)


def normalize():
    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    elements = data["kids"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    current_book = None
    current_section = None

    records = []

    for element in elements:
        element_type = element.get("type")
        page = element.get("page number")

        # --------------------------------------------------
        # HEADINGS
        # --------------------------------------------------

        if element_type == "heading":
            text = (element.get("content") or "").strip()

            if not text:
                continue

            # Detect book headings
            if text.upper().startswith("BOOK "):
                current_book = text
                current_section = None

            records.append({
                "id": element.get("id"),
                "page": page,
                "type": "heading",
                "book": current_book,
                "section": current_section,
                "text": text,
                "source": "complete_mahabharat.pdf",
            })

            continue

        # --------------------------------------------------
        # NORMAL PARAGRAPHS
        # --------------------------------------------------

        if element_type == "paragraph":
            text = (element.get("content") or "").strip()

            if not text:
                continue

            records.append({
                "id": element.get("id"),
                "page": page,
                "type": "paragraph",
                "book": current_book,
                "section": current_section,
                "text": text,
                "source": "complete_mahabharat.pdf",
            })

            continue

        # --------------------------------------------------
        # LIST ITEMS
        # --------------------------------------------------

        if element_type == "list":
            for item in element.get("list items", []):
                text = (item.get("content") or "").strip()

                if not text:
                    continue

                # Section headings often appear as list items
                upper = text.upper()

                if upper.startswith("SECTION "):
                    current_section = text

                    records.append({
                        "id": item.get("id"),
                        "page": item.get("page number", page),
                        "type": "section",
                        "book": current_book,
                        "section": current_section,
                        "text": text,
                        "source": "complete_mahabharat.pdf",
                    })

                else:
                    records.append({
                        "id": item.get("id"),
                        "page": item.get("page number", page),
                        "type": "list_item",
                        "book": current_book,
                        "section": current_section,
                        "text": text,
                        "source": "complete_mahabharat.pdf",
                    })

    # ------------------------------------------------------
    # WRITE JSONL
    # ------------------------------------------------------

    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("Normalization complete.")
    print("Input elements:", len(elements))
    print("Output records:", len(records))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    normalize()
