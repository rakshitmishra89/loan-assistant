import logging
from rag.retriever import retrieve as actual_retrieve, ingest_new_text
from perf.cache import retriever_cache, hash_key

logger = logging.getLogger(__name__)

def retrieve(query: str, k: int = 5) -> list:
    """Calls ChromaDB but checks the cache first for speed!"""
    try:
        # 1. Check Cache
        key = hash_key(query)
        cached_result = retriever_cache.get(key)
        if cached_result:
            logger.info("RAG CACHE HIT")
            return cached_result
            
        # 2. If not in cache, do the heavy search
        logger.info("RAG CACHE MISS - Searching database...")
        results = actual_retrieve(query, k)
        
        # 3. Save to cache for next time
        retriever_cache.set(key, results)
        return results
        
    except Exception as e:
        logger.error(f"RAG Error: {e}", exc_info=True)
        return []


def add_document(text: str, filename: str) -> int:
    """Ingests new text and flushes the cache so the bot instantly learns it."""
    
    # 1. Clear the retriever cache to prevent stale answers
    with retriever_cache.lock:
        retriever_cache.store.clear()
        logger.info("RAG Cache cleared due to new document upload.")
        
    # 2. Ingest the text
    return ingest_new_text(text, filename)
