from backend.adapters import rag_adapter

def process(user_message: str) -> dict:
    """
    In a fully autonomous setup, an LLM would write the search query.
    For speed, we will pass the user's message directly to our RAG adapter.
    """
    # Ask the RAG adapter to search ChromaDB
    chunks = rag_adapter.retrieve(user_message, k=4)
    
    if chunks:
        return {"used_rag": True, "chunks": chunks}
    else:
        return {"used_rag": False, "chunks": []}