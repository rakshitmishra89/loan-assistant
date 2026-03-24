import json
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

# We use JSON format so the AI always replies with computer-readable data
llm = ChatOllama(model="mistral", temperature=0.0, format="json")

def process(user_message: str, current_state: dict) -> dict:
    prompt = """
    You are an Intake Agent for a bank. Read the user's message and extract financial data.
    Current known data: {current_state}
    User Message: {message}
    
    Respond ONLY in JSON format with these exact keys:
    "loan_amount": (number or null),
    "income_monthly": (number or null),
    "tenure_months": (number or null),
    "age": (number or null),
    "missing_fields": (list of strings for what is still needed, e.g., ["income_monthly", "age"])
    """
    
    chain = PromptTemplate.from_template(prompt) | llm
    response = chain.invoke({"current_state": json.dumps(current_state), "message": user_message})
    
    try:
        # Strip markdown formatting that Mistral sometimes adds
        cleaned_content = response.content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
            
        return json.loads(cleaned_content.strip())
    except:
        return {"loan_amount": None, "income_monthly": None, "missing_fields": ["loan_amount", "income_monthly", "age", "tenure_months"]}