import os
from dotenv import load_dotenv
from google import genai
from ingestion.pdf_parser import extract_pdf
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenAIClassifier


load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"))
SUMMARY_MODEL = "gemini-3.5-flash-lite"
DECISION_MODEL = "gemini-decision-1"
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-3.8-flash")

_paper_cache = {}  # source path -> full text


def load_full_paper(path: str) -> str:
    if path not in _paper_cache:
        pages = extract_pdf(path)
        _paper_cache[path] = "\n\n".join(
            f"[Page {p.metadata['page_label']}]\n{p.page_content}" for p in pages
        )
    return _paper_cache[path]


def answer_with_summary(query: str, results: list) -> str:
    """results = output of retrieve_document: [(Document, score), ...]"""
    doc, score = results[0]  # top hit for now

    full_text = load_full_paper(doc.metadata["source"])

    prompt = f"""You are a research assistant and an expert in AI research! Your client is interested in a topic 
        (USER QUERY) and a relevant AI research paper was found (RETRIEVED PASSAGE):

USER QUERY:
{query}

RETRIEVED PASSAGE (page {doc.metadata['page_label']} of {doc.metadata['filename']}):
{doc.page_content}

FULL PAPER TEXT:
{full_text}

Respond in two parts:
1. **Paper summary**: a brief summary (3-4 sentences) of the entire paper, 
    written for a non-technical audience.
2. **Relevance to your query**: explain how the retrieved passage relates to the
   user's query, and add one useful related point from elsewhere in the paper
   that helps answer it. Base everything only on the text provided."""

    response = client.models.generate_content(model=SUMMARY_MODEL, contents=prompt)
    return response.text


def get_llm_summarizer(model=SUMMARY_MODEL):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        api_key = os.getenv("GEMINI_API_KEY")
    )

def get_llm_decider(model=DECISION_MODEL):
    return GoogleGenAIClassifier(
        model=model,
        api_key = os.getenv("GEMINI_API_KEY")
    )