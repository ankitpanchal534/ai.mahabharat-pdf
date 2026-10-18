import json
from collections import Counter
from pathlib import Path


JSON_FILE = Path(
    "data/extracted/complete_mahabharat/complete_mahabharat.json"
)


def main():
    with JSON_FILE.open("r", encoding="utf-8") as f:
        document = json.load(f)

    kids = document["kids"]

    print("=" * 70)
    print("DOCUMENT")
    print("=" * 70)

    print(f"File: {document['file name']}")
    print(f"Pages: {document['number of pages']:,}")
    print(f"Elements: {len(kids):,}")

    # ---------------------------------------------------------
    # Element types
    # ---------------------------------------------------------

    type_counts = Counter(
        item.get("type")
        for item in kids
    )

    print("\n" + "=" * 70)
    print("ELEMENT TYPES")
    print("=" * 70)

    for element_type, count in type_counts.most_common():
        print(f"{element_type:15} {count:,}")

    # ---------------------------------------------------------
    # Text quality
    # ---------------------------------------------------------

    text_elements = [
        item
        for item in kids
        if item.get("type") in {"paragraph", "heading", "caption"}
        and item.get("content")
    ]

    total_chars = 0
    null_chars = 0
    empty_elements = 0

    for item in text_elements:
        text = item.get("content", "")

        total_chars += len(text)
        null_chars += text.count("\x00")

        if not text.strip():
            empty_elements += 1

    print("\n" + "=" * 70)
    print("TEXT QUALITY")
    print("=" * 70)

    print(f"Text elements:       {len(text_elements):,}")
    print(f"Characters:          {total_chars:,}")
    print(f"NULL characters:     {null_chars:,}")
    print(f"Empty text elements: {empty_elements:,}")

    if total_chars:
        percentage = (null_chars / total_chars) * 100
        print(f"NULL percentage:     {percentage:.4f}%")

    # ---------------------------------------------------------
    # Pages
    # ---------------------------------------------------------

    pages = Counter()

    for item in text_elements:
        page = item.get("page number")

        if page is not None:
            pages[page] += 1

    print("\n" + "=" * 70)
    print("PAGE COVERAGE")
    print("=" * 70)

    print(f"Pages containing text: {len(pages):,}")

    if pages:
        print(f"First page:           {min(pages)}")
        print(f"Last page:            {max(pages)}")

    # ---------------------------------------------------------
    # Headings
    # ---------------------------------------------------------

    headings = [
        item
        for item in kids
        if item.get("type") == "heading"
    ]

    print("\n" + "=" * 70)
    print(f"HEADINGS: {len(headings):,}")
    print("=" * 70)

    for heading in headings[:100]:
        page = heading.get("page number")
        level = heading.get("heading level")
        content = heading.get("content", "").replace("\x00", "")

        print(
            f"[page {page}] "
            f"[level {level}] "
            f"{content!r}"
        )


if __name__ == "__main__":
    main()