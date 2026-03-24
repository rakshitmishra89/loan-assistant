from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="mistral", temperature=0.2)

def process(user_message: str, tool_results: dict, rag_data: dict, chat_history: str) -> tuple[str, dict]:
    prompt = """
    You are the Final Decision Agent for a bank.
    
    Conversation History: {history}
    User's Latest Request: {message}
    Financial Math Results: {tools}
    Bank Policies: {rag}
    
    Write a polite, professional response explaining the outcome. 
    IMPORTANT: You must format ALL monetary values in Indian Rupees (₹), never use Dollars.
    Look carefully at the Financial Math Results. 
    If 'is_eligible' is False OR the 'risk_band' is HIGH, you MUST decline the loan and explain the specific reason why. 
    If 'is_eligible' is True AND the 'risk_band' is LOW or MEDIUM, you may approve it.
    """
    
    chain = PromptTemplate.from_template(prompt) | llm
    reply = chain.invoke({
        "history": chat_history,
        "message": user_message, 
        "tools": tool_results, 
        "rag": rag_data.get("chunks", [])
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
    else:
        status = "APPROVE"
        reason = f"Risk band is {risk_band} and eligibility checks passed."
        
    decision = {
        "status": status,
        "reasoning": [reason],
        "confidence": 0.9
    }
    
    return reply, decision