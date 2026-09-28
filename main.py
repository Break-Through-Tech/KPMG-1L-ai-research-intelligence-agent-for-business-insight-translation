from ingestion.pdf_parser import extract_all_pdfs

documents = extract_all_pdfs('data')
print(documents)