"""
guardrails_adapter.py
=====================
Adapter layer between the Orchestrator and the Guardrails Agent.
Provides a clean interface and maps outputs to orchestrator's expected format.
"""

import logging
from guardrails.guardrails import (
    moderate_input as gi_moderate_input,
    moderate_output as gi_moderate_output,
    get_safe_response as gi_get_safe_response,
    detect_intent_hints as gi_detect_intent_hints
)

logger = logging.getLogger(__name__)


def moderate_input(text: str) -> dict:
    """
    Connects the Orchestrator to your real Guardrails Agent.
    Also includes intent hints for better routing decisions.
    """
    result = gi_moderate_input(text)
    
    # Map your guardrails output to what the orchestrator expects
    action = "BLOCK" if not result["allowed"] else "ALLOW"
    category_list = [result["category"]] if result["category"] != "clean" else []
    
    # Get intent hints for additional context
    intent_hints = gi_detect_intent_hints(text)
    
    return {
        "action": action,
        "categories": category_list,
        "redacted_text": result.get("redacted_text", text),
        "intent_hints": intent_hints  # Additional context for orchestrator
    }


def moderate_output(text: str) -> dict:
    """Checks the LLM's output before showing the user"""
    result = gi_moderate_output(text)
    
    action = "BLOCK" if not result["allowed"] else "ALLOW"
    category_list = [result["category"]] if result["category"] != "clean" else []
    
    return {
        "action": action,
        "categories": category_list,
        "safe_text": result.get("safe_text", text)
    }


def get_safe_response(category: str) -> str:
    """
    Gets the appropriate safe response message for a blocked category.
    This is shown to the user when their input is blocked.
    """
    return gi_get_safe_response(category)


def get_intent_hints(text: str) -> dict:
    """
    Get intent hints for a message using LLM-based analysis.
    Provides semantic understanding of user intent.
    
    Returns:
        {
            "is_security_threat": bool,  # Is this a prompt injection/hack attempt?
            "is_financial": bool,        # Contains loan/financial context
            "is_policy_query": bool,     # Asking about policies/rules
            "is_calculation": bool,      # Wants something calculated
            "threat_reason": str,        # Why it's a threat (if applicable)
            "confidence": float,         # 0.0 to 1.0
            "analysis_method": str       # "llm" or "regex_fallback"
        }
    """
    return gi_detect_intent_hints(text)
