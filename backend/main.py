# backend/main.py
import os
import time
import logging
from collections import defaultdict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Request

from backend.schemas import ChatRequest, ChatResponse, RagQueryRequest, RagQueryResponse
from backend.orchestrator import handle_chat
from backend.adapters import rag_adapter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Loan Approval & Credit Risk Assistant API")

# Simple in-memory rate limiter
class SimpleRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str, limit: int = 10, window: int = 60) -> bool:
        """Check if request is allowed based on rate limit."""
        now = time.time()
        # Clean old requests outside the window
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < window]
        
        if len(self.requests[client_ip]) >= limit:
            return False
        
        self.requests[client_ip].append(now)
        return True

rate_limiter = SimpleRateLimiter()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    # Simple rate limiting
    client_ip = req.client.host if req.client else "unknown"
    if not rate_limiter.is_allowed(client_ip, limit=10, window=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    
    try:
        logger.info(f"Chat request received for session: {request.session_id}")
        # Pass the request to your autonomous orchestrator
        response = await handle_chat(
            session_id=request.session_id, 
            message=request.message,
            metadata=request.metadata
        )
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=RagQueryResponse)
async def rag_query_endpoint(request: RagQueryRequest):
    """Direct endpoint for Streamlit debugging and evaluation notebooks."""
    try:
        chunks = rag_adapter.retrieve(request.query, request.k)
        return RagQueryResponse(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), req: Request = None):
    # Simple rate limiting for uploads
    client_ip = req.client.host if req and req.client else "unknown"
    if not rate_limiter.is_allowed(client_ip, limit=5, window=60):
        raise HTTPException(status_code=429, detail="Upload rate limit exceeded. Try again later.")
    
    try:
        logger.info(f"Document upload request: {file.filename}")
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # Pass it to the RAG Adapter to store in ChromaDB
        chunks_added = rag_adapter.add_document(text_content, file.filename)
        
        logger.info(f"Document {file.filename} ingested with {chunks_added} chunks")
        return {"status": "success", "filename": file.filename, "chunks_added": chunks_added}
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
