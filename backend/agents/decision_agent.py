import os
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Environment variable configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

llm = ChatOllama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)


def _is_emi_calculation(user_message: str, tool_results: dict) -> bool:
    """Check if this is primarily an EMI calculation request."""
    emi_keywords = ["emi", "calculate", "monthly payment", "installment"]
    message_lower = user_message.lower()
    has_emi_request = any(kw in message_lower for kw in emi_keywords)
    has_emi_result = tool_results.get("emi", 0) > 0
    return has_emi_request and has_emi_result


def _is_policy_question(tool_results: dict, rag_data: dict) -> bool:
    """
    Determines if this is a policy/information question vs a loan application decision.
    Policy questions have RAG data but no meaningful tool results.
    """
    has_rag = rag_data.get("used_rag", False) and len(rag_data.get("chunks", [])) > 0
    has_tools = any([
        tool_results.get("emi"),
        tool_results.get("is_eligible") is not None,
        tool_results.get("risk_band") not in [None, "UNKNOWN"]
    ])
    
    # If we have RAG data but no meaningful tool calculations, it's a policy question
    return has_rag and not has_tools

def _is_off_topic_query(user_message: str, rag_data: dict) -> bool:
    """
    Dynamically detect if a query is off-topic by checking:
    1. RAG returned no relevant chunks (or very low relevance)
    2. The query doesn't match loan/finance keywords
    
    This provides a fallback for when the LLM intent detection misses off-topic queries.
    """
    # Check if RAG found any relevant content
    chunks = rag_data.get("chunks", [])
    has_relevant_rag = len(chunks) > 0
    
    # If RAG found content, check the relevance scores if available
    if has_relevant_rag:
        # Check if any chunk has good relevance (some RAG implementations include scores)
        # For now, we assume if chunks are returned, they might be relevant
        pass
    
    # Financial/loan keywords that indicate on-topic queries
    financial_keywords = [
        "loan", "emi", "interest", "credit", "cibil", "eligibility",
        "income", "salary", "lakh", "crore", "rupee", "rs", "inr",
        "tenure", "mortgage", "repayment", "principal", "rate",
        "fee", "charge", "document", "kyc", "apply", "application",
        "borrow", "bank", "financial", "payment", "installment"
    ]
    
    message_lower = user_message.lower()
    has_financial_context = any(kw in message_lower for kw in financial_keywords)
    
    # If no RAG results AND no financial keywords, likely off-topic
    if not has_relevant_rag and not has_financial_context:
        return True
    
    return False



def process(user_message: str, tool_results: dict, rag_data: dict, chat_history: str, intent_hints: dict = None) -> tuple[str, dict]:
    """
    DECISION AGENT
    --------------
    Handles four types of responses:
    1. Off-Topic Queries: Questions unrelated to loans/banking
    2. EMI Calculations: Direct calculation results with minimal LLM involvement
    3. Policy Questions: Uses RAG data to answer informational queries
    4. Loan Decisions: Uses tool results to make approval/rejection decisions
    
    Args:
        user_message: The user's query
        tool_results: Results from financial tools (EMI, eligibility, risk)
        rag_data: Retrieved policy documents
        chat_history: Conversation history
        intent_hints: Optional intent analysis from guardrails (is_off_topic, is_financial, etc.)
    """
    
    # CHECK 0: Is this an off-topic query?
    # First check intent_hints from guardrails if available
    is_off_topic_from_llm = False
    off_topic_reason = None
    
    if intent_hints:
        is_off_topic_from_llm = intent_hints.get("is_off_topic", False)
        off_topic_reason = intent_hints.get("off_topic_reason", "")
    
    # Also do a dynamic check based on RAG results and content
    is_off_topic_dynamic = _is_off_topic_query(user_message, rag_data)
    
    # Combine both checks - if either flags it as off-topic
    if is_off_topic_from_llm or is_off_topic_dynamic:
        reply = (
            "I'm a Loan Assistant and can only help with loan-related queries such as:\n"
            "- Loan applications and eligibility\n"
            "- EMI calculations\n"
            "- Interest rates and fees\n"
            "- Policy and documentation questions\n\n"
            "Your question appears to be about something else"
        )
        if off_topic_reason:
            reply += f" ({off_topic_reason})"
        reply += ". Please ask me about loans or banking services!"
        
        decision = {
            "status": "OFF_TOPIC",
            "reasoning": [f"Query unrelated to loans/banking: {off_topic_reason or 'No financial context detected'}"],
            "confidence": 0.9
        }
        return reply, decision
    
    # CHECK 1: Is this an EMI calculation request? Handle with direct response (no hallucination)
    if _is_emi_calculation(user_message, tool_results):
        emi = tool_results.get("emi", 0)
        principal = tool_results.get("principal", 0)
        tenure = tool_results.get("tenure_used", 36)
        interest_rate = tool_results.get("interest_rate_used", 12.5)
        
        # Direct factual response - no LLM generation to avoid hallucination
        reply = f"""Based on your request, here are the EMI calculation results:

**Loan Details:**
- Principal Amount: Rs. {principal:,.2f}
- Interest Rate: {interest_rate}% per annum
- Tenure: {tenure} months ({tenure // 12} years {tenure % 12} months)

**Calculated EMI: Rs. {emi:,.2f} per month**

Total Amount Payable: Rs. {(emi * tenure):,.2f}
Total Interest: Rs. {(emi * tenure - principal):,.2f}

Note: This is a standard reducing balance EMI calculation. Actual EMI may vary based on processing fees and other charges."""
        
        decision = {
            "status": "CALCULATION_COMPLETE",
            "reasoning": [f"EMI calculated: Rs. {emi:,.2f}/month for {tenure} months at {interest_rate}%"],
            "confidence": 1.0  # 100% confidence as this is math, not prediction
        }
        return reply, decision
    
    # CHECK 2: Is this a policy question?
    is_policy = _is_policy_question(tool_results, rag_data)
    
    if is_policy:
        # POLICY QUESTION MODE: Answer based on RAG context - STRICT NO HALLUCINATION
        policy_prompt = """
        You are a STRICT Bank Policy Assistant. You ONLY answer from the provided context.
        
        User's Question: {message}
        
        POLICY DOCUMENTS (Your ONLY source of truth):
        {rag}
        
        CRITICAL RULES - FOLLOW EXACTLY:
        1. ONLY use information that is EXPLICITLY stated in the POLICY DOCUMENTS above.
        2. If the answer is NOT in the context, respond EXACTLY with: "I'm a Loan Assistant and can only help with loan-related queries. I don't have information about that topic. Please ask me about loans, EMI, interest rates, eligibility, or other banking services!"
        3. If the answer is NOT in the context, respond: "I don't have information about that in our policy documents. Please contact customer support for more details."
        4. DO NOT invent, assume, or add ANY information not in the context.
        5. DO NOT mention websites, portals, phone numbers, or contact details unless they are in the context.
        6. Quote specific policy text when possible.
        7. Keep your response concise and factual.
        8. Use Rs. for Indian Rupee amounts.
        """
        
        # Format RAG chunks for the prompt
        rag_context = "\n\n".join([
            f"--- Policy Excerpt ---\n{chunk.get('text', '')}"
            for chunk in rag_data.get("chunks", [])
        ])
        
        # If no RAG context or very limited context, this is likely off-topic
        if not rag_context or len(rag_context.strip()) < 50:
            reply = (
                "I'm a Loan Assistant and can only help with loan-related queries such as:\n"
                "- Loan applications and eligibility\n"
                "- EMI calculations\n"
                "- Interest rates and fees\n"
                "- Policy and documentation questions\n\n"
                "I don't have information about that topic. Please ask me about loans or banking services!"
            )
            decision = {
                "status": "OFF_TOPIC",
                "reasoning": ["No relevant policy information found for this query"],
                "confidence": 0.85
            }
            return reply, decision

        chain = PromptTemplate.from_template(policy_prompt) | llm
        reply = chain.invoke({
            "message": user_message,
            "rag": rag_context if rag_context else "No relevant policy information found."
        }).content
        
        decision = {
            "status": "INFO_PROVIDED",
            "reasoning": ["Answered policy/information question using RAG"],
            "confidence": 0.85
        }
        
        return reply, decision
    
    # LOAN DECISION MODE: Make approval/rejection decision
    loan_prompt = """
    You are the Final Decision Agent for a bank.
    
    Conversation History: {history}
    User's Latest Request: {message}
    Financial Math Results: {tools}
    Bank Policies: {rag}
    
    Write a polite, professional response explaining the outcome. 
    IMPORTANT: You must format ALL monetary values in Indian Rupees (Rs.) or as specified.
    Look carefully at the Financial Math Results. 
    If 'is_eligible' is False OR the 'risk_band' is HIGH, you MUST decline the loan and explain the specific reason why. 
    If 'is_eligible' is True AND the 'risk_band' is LOW or MEDIUM, you may approve it.
    Include the calculated EMI if available.
    """
    
    rag_context = "\n".join([
        chunk.get('text', '')[:500] 
        for chunk in rag_data.get("chunks", [])[:3]
    ])
    
    chain = PromptTemplate.from_template(loan_prompt) | llm
    reply = chain.invoke({
        "history": chat_history,
        "message": user_message, 
        "tools": tool_results, 
        "rag": rag_context if rag_context else "No specific policies retrieved."
    }).content
    
    # Smarter decision logic combining Eligibility Tool AND Risk Tool
    is_eligible = tool_results.get("is_eligible", True)
    risk_band = tool_results.get("risk_band", "UNKNOWN")
    
    if not is_eligible:
        status = "REJECT"
        reason = "Failed basic age or income eligibility rules."
    elif risk_band == "HIGH":
        status = "REJECT"
        reason = f"Calculated Risk Band is {risk_band}"
    elif risk_band in ["LOW", "MEDIUM"]:
        status = "APPROVE"
        reason = f"Risk band is {risk_band} and eligibility checks passed."
    else:
        # No tool data available yet
        status = "PENDING"
        reason = "Awaiting complete financial information for decision."
        
    decision = {
        "status": status,
        "reasoning": [reason],
        "confidence": 0.9 if status in ["APPROVE", "REJECT"] else 0.5
    }
    
    return reply, decision
