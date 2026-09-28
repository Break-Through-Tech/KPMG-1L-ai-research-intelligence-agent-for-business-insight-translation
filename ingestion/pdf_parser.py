from pathlib import Path
from pypdf import PdfReader


# Extracts one pdf into a dictionary with metadata + list of pages
def extract_pdf(pdf_path: Path) -> dict:

    reader = PdfReader(pdf_path)
    pages = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        pages.append({
            "page_number": page_index+1,
            "text": text.strip()
        })


    document = {
        "document_id": pdf_path.stem,
        "filename": pdf_path.name,
        "pages": pages
    }
    return document

# Extracts all pdfs in a directory
def extract_all_pdfs(pdf_directory: str) -> list[dict]:
    documents = []

    for pdf_path in Path(pdf_directory).glob("*pdf"):
        print(f"Extracting {pdf_path.name}")
        documents.append(extract_pdf(pdf_path))

    return documents


