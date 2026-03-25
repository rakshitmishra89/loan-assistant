import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama 
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. CONFIGURATION VIA ENVIRONMENT VARIABLES
# ---------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
DB_DIR = os.getenv("CHROMA_DB_PATH", os.path.join(os.path.dirname(__file__), "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_K = int(os.getenv("RAG_DEFAULT_K", "5"))

# ---------------------------------------------------------
# 2. LAZY-LOADED INSTANCES (avoid import-time initialization)
# ---------------------------------------------------------
_llm = None
_vector_store = None
_embeddings = None


def _get_llm():
    """Lazy-load the LLM instance."""
    global _llm
    if _llm is None:
        logger.info(f"Initializing Ollama LLM with model: {OLLAMA_MODEL}")
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE
        )
    return _llm


def _get_embeddings():
    """Lazy-load the embeddings model."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"Initializing embeddings with model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_vector_store():
    """Lazy-load the vector store."""
    global _vector_store
    if _vector_store is None:
        logger.info(f"Initializing ChromaDB at: {DB_DIR}")
        _vector_store = Chroma(
            persist_directory=DB_DIR, 
            embedding_function=_get_embeddings()
        )
    return _vector_store


# ---------------------------------------------------------
# 3. CORE DELIVERABLE: THE RETRIEVAL FUNCTION
# ---------------------------------------------------------
def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    """
    Searches the local ChromaDB and returns the top k relevant chunks.
    
    Args:
        query: The search query string
        k: Number of top results to return (default from env or 5)
    
    Returns:
        List of dicts with text, score, source, and section
    """
    try:
        vector_store = _get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "text": doc.page_content,
                "score": float(score), 
                "source": doc.metadata.get("source", "master_policy_doc.txt"),
                "section": doc.metadata.get("section", "General")
            })
        return formatted_results
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        return []


# ---------------------------------------------------------
# 4. FULL RAG GENERATION PIPELINE
# ---------------------------------------------------------
def generate_rag_answer(query: str) -> dict:
    """Generate an answer using RAG pipeline."""
    try:
        chunks = retrieve(query, k=DEFAULT_K) 
        context_text = "\n\n".join([f"Context Chunk:\n{c['text']}" for c in chunks])

        prompt_template = """
        You are an incredibly strict Loan and Credit Risk Compliance Auditor. 
        You evaluate queries strictly against the provided CONTEXT. 
        You have zero imagination. You never invent website portals, login steps, or external advice.

        CRITICAL RULES:
        1. If the exact answer is not in the context, your ONLY output must be: "REJECTED: Information not found in company policy."
        2. Read the time limits carefully. If a rule says "only after X months", and the user is at Y months (where Y < X), the action is FORBIDDEN.
        
        CONTEXT:
        {context}
        
        USER QUESTION:
        {question}
        
        You MUST format your answer exactly like this:
        POLICY CITED: [Quote the exact sentence from the context, or write 'None']
        DECISION: [Approved / Denied / Cannot Determine]
        EXPLANATION: [One factual sentence explaining why, with NO extra advice]
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        llm = _get_llm()
        chain = prompt | llm
        
        response = chain.invoke({"context": context_text, "question": query})
        
        return {
            "answer": response.content,
            "chunks_used": len(chunks)
        }
    except Exception as e:
        logger.error(f"RAG generation error: {e}", exc_info=True)
        return {
            "answer": "Error processing your request. Please try again.",
            "chunks_used": 0
        }


def ingest_new_text(text: str, filename: str) -> int:
    """Chunks a newly uploaded text file and adds it to the existing ChromaDB."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        # Add metadata so we know where it came from
        metadatas = [{"source": filename, "section": "User Uploaded"} for _ in chunks]
        
        # Add to the existing vector store
        vector_store = _get_vector_store()
        vector_store.add_texts(texts=chunks, metadatas=metadatas)
        
        logger.info(f"Ingested {len(chunks)} chunks from {filename}")
        return len(chunks)
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        return 0
