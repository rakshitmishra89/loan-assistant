import os
import re
import json
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Environment variable configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

# We use JSON format so the AI always replies with computer-readable data
llm = ChatOllama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE, format="json")


# ============================================================
#  SECTION 1 — LLM-BASED INTELLIGENT ENTITY EXTRACTION
#  Uses AI to understand natural language and extract values
# ============================================================

def _extract_entities_with_llm(message: str) -> dict:
    """
    LLM-BASED ENTITY EXTRACTION
    ---------------------------
    Uses the LLM to intelligently extract loan-related entities from
    natural language. This is dynamic and can understand various
    phrasings like:
    - "5-year time period" -> tenure_months: 60
    - "repay over 5 years" -> tenure_months: 60
    - "I earn 80000 per month" -> income_monthly: 80000
    - "20 lakhs loan" -> loan_amount: 2000000
    
    The LLM understands context and semantics, not just patterns.
    """
    
    extraction_prompt = """You are an intelligent entity extraction agent for a loan application system.
Your task is to analyze the user's message and extract all loan-related information.

USER MESSAGE: {message}

EXTRACTION RULES:
1. **loan_amount**: The amount of money the user wants to borrow
   - Convert Indian formats: "20 lakhs" = 2000000, "1 crore" = 10000000, "50k" = 50000
   - Look for phrases like: "loan of X", "borrow X", "need X", "want X loan"

2. **income_monthly**: The user's monthly income/salary
   - If annual income is given, divide by 12
   - Look for: "earn X per month", "salary X", "monthly income X", "I earn X"
   - Convert: "80000 per month" = 80000, "6 lakh per annum" = 50000

3. **tenure_months**: How long they want to repay (ALWAYS in months)
   - Convert years to months: "5 years" = 60, "5-year" = 60, "5 year period" = 60
   - Look for: "X years", "X-year", "over X years", "repay in X years", "X year time period", "tenure X", "for X years"
   - If months given directly, use as-is: "36 months" = 36

4. **age**: The user's age in years (must be 18-100)
   - Look for: "I am X years old", "age X", "aged X", "X years old"

5. **credit_score**: CIBIL/credit score (must be 300-900)
   - Look for: "credit score X", "CIBIL X", "score is X"

6. **interest_rate**: Interest rate percentage (must be 1-30)
   - Look for: "X% interest", "rate of X%", "at X%"

IMPORTANT:
- Return ONLY the values you can confidently extract from the message
- Use null for any field you cannot determine
- For tenure, ALWAYS convert to months (years * 12)
- For loan_amount and income, convert lakhs/crores to actual numbers

Respond ONLY in this JSON format (no explanation):
{{
    "loan_amount": <number or null>,
    "income_monthly": <number or null>,
    "tenure_months": <number or null>,
    "age": <number or null>,
    "credit_score": <number or null>,
    "interest_rate": <number or null>
}}"""

    try:
        chain = PromptTemplate.from_template(extraction_prompt) | llm
        response = chain.invoke({"message": message})
        
        # Parse the JSON response
        cleaned_content = response.content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        
        result = json.loads(cleaned_content.strip())
        logger.info(f"LLM extraction result: {result}")
        
        # Validate and clean the extracted values
        validated = {}
        
        if result.get("loan_amount") and isinstance(result["loan_amount"], (int, float)) and result["loan_amount"] > 0:
            validated["loan_amount"] = float(result["loan_amount"])
        
        if result.get("income_monthly") and isinstance(result["income_monthly"], (int, float)) and result["income_monthly"] > 0:
            validated["income_monthly"] = float(result["income_monthly"])
        
        if result.get("tenure_months") and isinstance(result["tenure_months"], (int, float)) and result["tenure_months"] > 0:
            validated["tenure_months"] = int(result["tenure_months"])
        
        if result.get("age") and isinstance(result["age"], (int, float)) and 18 <= result["age"] <= 100:
            validated["age"] = int(result["age"])
        
        if result.get("credit_score") and isinstance(result["credit_score"], (int, float)) and 300 <= result["credit_score"] <= 900:
            validated["credit_score"] = int(result["credit_score"])
        
        if result.get("interest_rate") and isinstance(result["interest_rate"], (int, float)) and 1 <= result["interest_rate"] <= 30:
            validated["interest_rate"] = float(result["interest_rate"])
        
        return validated
        
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}, falling back to regex")
        return _extract_values_regex_fallback(message)


def _extract_values_regex_fallback(message: str) -> dict:
    """
    FALLBACK REGEX EXTRACTION
    -------------------------
    Used only if LLM extraction fails. Basic pattern matching.
    """
    extracted = {}
    msg_lower = message.lower()
    
    # Loan amount - basic patterns
    loan_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b', msg_lower)
    if loan_match:
        extracted["loan_amount"] = float(loan_match.group(1)) * 100000
    
    crore_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b', msg_lower)
    if crore_match:
        extracted["loan_amount"] = float(crore_match.group(1)) * 10000000
    
    # Income - basic patterns  
    income_match = re.search(r'(?:earn|salary|income).*?(\d+(?:,\d+)*)', msg_lower)
    if income_match:
        income_val = income_match.group(1).replace(',', '')
        extracted["income_monthly"] = float(income_val)
    
    # Tenure - more flexible patterns
    tenure_patterns = [
        (r'(\d+)[\s\-]*year', 'years'),
        (r'(\d+)\s*months?', 'months'),
    ]
    for pattern, unit in tenure_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            val = int(match.group(1))
            extracted["tenure_months"] = val * 12 if unit == 'years' else val
            break
    
    # Age
    age_match = re.search(r'(?:i\s*am|age|aged)\s*(\d+)', msg_lower)
    if age_match:
        age_val = int(age_match.group(1))
        if 18 <= age_val <= 100:
            extracted["age"] = age_val
    
    # Credit score
    credit_match = re.search(r'(?:credit|cibil|score).*?(\d{3})\b', msg_lower)
    if credit_match:
        score = int(credit_match.group(1))
        if 300 <= score <= 900:
            extracted["credit_score"] = score
    
    return extracted


def classify_intent(user_message: str) -> str:
    """
    INTENT CLASSIFICATION
    ---------------------
    Determines which type of query the user is making:
    - "loan_application": User wants to apply for a loan (needs financial data collection)
    - "policy_question": User is asking about policies, rates, eligibility rules (use RAG)
    - "calculation": User wants EMI calculation or eligibility check (use tools)
    - "general": General greeting or unrelated query
    """
    intent_prompt = """
    You are an Intent Classifier for a bank assistant. Analyze the user's message and classify it.
    
    User Message: {message}
    
    CLASSIFICATION RULES:
    1. "loan_application" - User explicitly wants to APPLY for a loan, start an application, or provide their financial details for a loan
       Examples: "I want to apply for a loan", "I need a home loan of 50 lakhs", "Start my loan application"
    
    2. "policy_question" - User is ASKING ABOUT policies, rules, interest rates, eligibility criteria, fees, penalties, credit cards, or any informational question about bank products
       Examples: "What is the interest rate?", "What are the eligibility criteria?", "Tell me about foreclosure policy", "What documents are needed?", "What is the processing fee?", "How does EMI calculation work?"
    
    3. "calculation" - User wants to CALCULATE something specific like EMI, check eligibility with specific numbers they provide
       Examples: "Calculate my EMI for 10 lakhs at 12% for 5 years", "Am I eligible for a loan if my salary is 50000?"
    
    4. "general" - Greetings, thanks, general conversation, or unclear intent
       Examples: "Hello", "Thank you", "What can you do?"
    
    Respond ONLY in JSON format:
    {{"intent": "loan_application" | "policy_question" | "calculation" | "general"}}
    """
    
    chain = PromptTemplate.from_template(intent_prompt) | llm
    response = chain.invoke({"message": user_message})
    
    try:
        cleaned_content = response.content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        result = json.loads(cleaned_content.strip())
        return result.get("intent", "general")
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to policy_question")
        # Default to policy_question if classification fails (safer than assuming loan application)
        return "policy_question"


def process(user_message: str, current_state: dict) -> dict:
    """
    INTAKE AGENT WITH INTENT-BASED ROUTING
    --------------------------------------
    1. FIRST: Extracts values using LLM-based intelligent analysis
    2. THEN: Classifies the user's intent
    3. FINALLY: Routes to appropriate flow
    
    This approach ensures:
    - Natural language understanding (not just regex patterns)
    - "5-year time period", "repay over 5 years", etc. all work
    - Tenure in years is auto-converted to months
    - Annual income is auto-converted to monthly income
    - Indian number formats (lakhs, crores) are properly parsed
    """
    
    # Step 1: LLM-BASED EXTRACTION (intelligent, understands natural language)
    llm_extracted = _extract_entities_with_llm(user_message)
    logger.info(f"LLM extraction result: {llm_extracted}")
    
    # Step 2: Classify the intent
    intent = classify_intent(user_message)
    logger.info(f"Intent classified as: {intent}")
    
    # Step 3: Merge LLM-extracted values with current state
    # New values from LLM override old values
    merged_state = {
        "loan_amount": llm_extracted.get("loan_amount") or current_state.get("loan_amount"),
        "income_monthly": llm_extracted.get("income_monthly") or current_state.get("income_monthly"),
        "tenure_months": llm_extracted.get("tenure_months") or current_state.get("tenure_months"),
        "age": llm_extracted.get("age") or current_state.get("age"),
        "credit_score": llm_extracted.get("credit_score") or current_state.get("credit_score"),
        "interest_rate": llm_extracted.get("interest_rate") or current_state.get("interest_rate") or 12.5,
    }
    
    # Step 4: For policy questions, general queries - skip data collection
    if intent in ["policy_question", "general"]:
        return {
            "intent": intent,
            **merged_state,
            "missing_fields": [],  # No fields required for policy questions
            "route_to": "rag" if intent == "policy_question" else "general"
        }
    
    # Step 5: For calculations - return merged state with tool routing
    if intent == "calculation":
        # Check what's still missing for calculation
        missing = []
        if not merged_state.get("loan_amount"):
            missing.append("loan_amount")
        if not merged_state.get("tenure_months"):
            missing.append("tenure (in years or months)")
        
        return {
            "intent": intent,
            **merged_state,
            "missing_fields": missing,
            "route_to": "tools" if not missing else "need_info"
        }
    
    # Step 6: For loan applications - check what fields are still needed
    required_fields = ["loan_amount", "income_monthly", "tenure_months", "age"]
    missing_fields = []
    
    for field in required_fields:
        if not merged_state.get(field):
            # Use friendly names for missing fields
            friendly_names = {
                "loan_amount": "loan amount",
                "income_monthly": "monthly income/salary",
                "tenure_months": "loan tenure (in years or months)",
                "age": "your age"
            }
            missing_fields.append(friendly_names.get(field, field))
    
    return {
        "intent": "loan_application",
        **merged_state,
        "missing_fields": missing_fields,
        "route_to": "loan_flow" if not missing_fields else "need_info"
    }


# ============================================================
#  SECTION 3 — DEMO / TEST
# ============================================================

if __name__ == "__main__":
    test_cases = [
        "I want a loan of 10 lakhs for 5 years",
        "My salary is 50000 per month",
        "I earn 6 lakh per annum",
        "Calculate EMI for 20L at 12% for 3 years",
        "I am 35 years old with credit score 750",
        "Need 1 crore loan, tenure 20 yrs, I earn 1.5L monthly",
        "What is the interest rate for home loan?",
        "loan amount 15 lacs, tenure 10 years, monthly salary 80k",
        # New test cases for natural language understanding
        "I want a home loan of 20 lakhs. I am 30 years old, and I earn 80000 per month and I want to repay over a 5-year time period. My credit score is 750",
        "I need 50 lakhs for 10-year period",
        "repay in 3 years with monthly salary of 1 lakh",
    ]
    
    print("=" * 70)
    print("  INTAKE AGENT - LLM-BASED ENTITY EXTRACTION TEST")
    print("=" * 70)
    
    for test in test_cases:
        print(f"\nInput: {test}")
        print("-" * 50)
        result = _extract_entities_with_llm(test)
        print(f"LLM Extracted: {result}")
        print("-" * 70)
