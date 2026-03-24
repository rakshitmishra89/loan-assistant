# backend/memory_store.py
from typing import Dict, Any

_sessions: Dict[str, Dict[str, Any]] = {}

def load(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "entities": {
                "income_monthly": None,
                "loan_amount": None,
                "tenure_months": None,
                "credit_score": None,
                "age": None
            },
            "summary": "Conversation started.\n"
        }
    return _sessions[session_id]

def save(session_id: str, state: Dict[str, Any], new_user_msg: str = "", new_ai_msg: str = ""):
    """Updates the state and maintains a running chat history summary."""
    if new_user_msg and new_ai_msg:
        # Keep a running log of the last few turns for Summary Memory
        current_summary = state.get("summary", "")
        updated_summary = current_summary + f"User: {new_user_msg}\nBank: {new_ai_msg}\n"
        
        # Prevent memory from blowing up by keeping only the last 1000 characters
        if len(updated_summary) > 1000:
            updated_summary = "..." + updated_summary[-1000:]
            
        state["summary"] = updated_summary
        
    _sessions[session_id] = state