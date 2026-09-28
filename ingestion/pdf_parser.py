from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


# Extracts one PDF into a collection of all its pages + data
def extract_pdf(pdf_path: str | Path) -> list[Document]:
    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)
    pages = []

    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        document = Document(
            page_content=text,
            metadata={
                "document_id": pdf_path.stem,
                "filename": pdf_path.name,
                "source": str(pdf_path),
                "page": page_index,
                "page_label": str(page_index + 1),
            },
        )

        pages.append(document)

    return pages

# Extracts all PDFs in a directory to a flat list of all their pages + metadata
def extract_all_pdfs(pdf_directory: str | Path) -> list[Document]:
    pdf_directory = Path(pdf_directory)
    documents = []

    for pdf_path in pdf_directory.glob('*pdf'):
        document = extract_pdf(pdf_path)
        documents.extend(document)

    return documents



