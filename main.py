from ingestion.pdf_parser import extract_all_pdfs
from ingestion.chunker import chunk_documents
from rag.rag_pipeline import store_documents, retrieve_document


documents = extract_all_pdfs('data')
print("✅ Extracted all documents")

chunks = chunk_documents(documents)
print("✅ Completed Chunking")
print(type(chunks))

print(len(chunks))

# Only run when storing new documents
store_documents(chunks)
#print(retrieve_document(query=input("Enter Query: "), k=1))
