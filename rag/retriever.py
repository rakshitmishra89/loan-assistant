import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama 
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------
# 1. SETUP LOCAL OLLAMA LLM (MISTRAL)
# ---------------------------------------------------------
print("Booting up local Mistral connection...")
llm = ChatOllama(
    model="mistral", # Change to "llama3.2:1b" if your computer runs out of RAM again
    temperature=0.0 
)

# ---------------------------------------------------------
# 2. SETUP LOCAL VECTOR STORE
# ---------------------------------------------------------
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)




# ---------------------------------------------------------
# 3. CORE DELIVERABLE: THE RETRIEVAL FUNCTION
# ---------------------------------------------------------
def retrieve(query: str, k: int = 5) -> list[dict]: # Increased k from 4 to 5
    """
    Searches the local ChromaDB and returns the top k relevant chunks.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "text": doc.page_content,
            "score": float(score), 
            "source": doc.metadata.get("source", "master_policy_document.txt"),
            "section": "General" 
        })
    return formatted_results

# ---------------------------------------------------------
# 4. FULL RAG GENERATION PIPELINE
# ---------------------------------------------------------
def generate_rag_answer(query: str) -> dict:
    # We now pull the top 5 chunks to ensure we don't miss trailing bullet points
    chunks = retrieve(query, k=5) 
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
    chain = prompt | llm
    
    response = chain.invoke({"context": context_text, "question": query})
    
    return {
        "answer": response.content,
        "chunks_used": len(chunks)
    }