import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path
import time


load_dotenv()  # Finds and loads API Key

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "sample_papers"

# Instantiates Embedding Model
def get_embedding_client() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs = {
            "device": "cpu"
        },
        encode_kwargs={
            "batch_size": 32,
            "normalize_embeddings": True,
        },
        show_progress = True
    )

def store_documents(documents: list[Document]) -> Chroma:
    embeddings = get_embedding_client()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory = PERSIST_DIR
    )

    batch_size = 128

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start+batch_size]

        vector_store.add_documents(batch)

        print(
            f"Stored {start+len(batch)}"
            f"of {len(documents)} chunks"
        )

    return vector_store


def retrieve_document(query: str, k: int=1):
    
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            "Vector database not found. "
            "Run store_documents() first."
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function = get_embedding_client(),
        persist_directory = PERSIST_DIR,
        create_collection_if_not_exists=False,
    )

    return vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=k
    )


