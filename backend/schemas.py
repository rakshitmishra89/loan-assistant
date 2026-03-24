# backend/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- Requests ---
class ChatRequest(BaseModel):
    session_id: str
    message: str
    uploaded_doc_ids: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}

class RagQueryRequest(BaseModel):
    query: str
    k: int = 4

# --- Nested Response Models ---
class DecisionModel(BaseModel):
    status: str = "NEED_MORE_INFO" # APPROVE|REJECT|MANUAL_REVIEW|NEED_MORE_INFO
    reasoning: List[str] = []
    confidence: float = 0.0

class ToolResultsModel(BaseModel):
    is_eligible: bool = True
    eligibility_reasons: List[str] = []
    emi: float = 0.0
    emi_burden_pct: float = 0.0
    risk_band: str = "UNKNOWN"
    tenure_used: int = 36
    principal: float = 0.0 

class RagChunkModel(BaseModel):
    text: str
    score: float
    source: str
    section: str

class RagMetadataModel(BaseModel):
    used: bool = False
    top_k: int = 0
    chunks: List[RagChunkModel] = []

class GuardrailsModel(BaseModel):
    input_action: str = "ALLOW"
    output_action: str = "ALLOW"
    categories: List[str] = []

class LatencyModel(BaseModel):
    retrieval: float = 0.0
    llm: float = 0.0
    end_to_end: float = 0.0

# --- Main Responses ---
class ChatResponse(BaseModel):
    session_id: str
    reply: str
    decision: DecisionModel
    collected_inputs: Dict[str, Any] = {}
    tool_results: ToolResultsModel
    rag: RagMetadataModel
    guardrails: GuardrailsModel
    agent_trace: List[Dict[str, Any]] = []
    latency_ms: LatencyModel

class RagQueryResponse(BaseModel):
    chunks: List[RagChunkModel]