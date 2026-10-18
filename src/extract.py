from pathlib import Path
# pyrefly: ignore [missing-import]
import opendataloader_pdf

INPUT_FILE = Path("data/raw/complete_mahabharat.pdf")
OUTPUT_DIR = Path("data/extracted/complete_mahabharat")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

opendataloader_pdf.convert(
    input_path=[str(INPUT_FILE)],
    output_dir=str(OUTPUT_DIR),
    format="json,markdown",
)

print(f"Extraction complete: {OUTPUT_DIR}")