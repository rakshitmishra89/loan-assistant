import sys
import os
# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from guardrails.guardrails import moderate_input as gi_moderate_input
from guardrails.guardrails import moderate_output as gi_moderate_output

def moderate_input(text: str) -> dict:
    """Connects the Orchestrator to your real Guardrails Agent"""
    result = gi_moderate_input(text)
    
    # Map your guardrails output to what the orchestrator expects
    action = "BLOCK" if not result["allowed"] else "ALLOW"
    category_list = [result["category"]] if result["category"] != "clean" else []
    
    return {
        "action": action,
        "categories": category_list,
        "redacted_text": result.get("redacted_text", text)
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