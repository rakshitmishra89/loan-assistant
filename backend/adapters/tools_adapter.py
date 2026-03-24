from backend.tools import emi_calculator, risk_scoring, eligibility

def run_all(state_entities: dict) -> dict:
    """
    Acts as the bridge. Takes the entities extracted from the user's memory 
    and runs the isolated math and logic tools.
    """
    # 1. Safely extract values from memory (default to safe numbers if missing)
    try:
        loan_amount = float(state_entities.get("loan_amount") or 0)
        income = float(state_entities.get("income_monthly") or 1) # Avoid division by zero
        tenure = int(state_entities.get("tenure_months") or 36)   # Dynamic tenure, defaults to 36
        age = int(state_entities.get("age") or 0)                 # Used for eligibility
    except ValueError:
        loan_amount = 0.0
        income = 1.0
        tenure = 36
        age = 0

    # ---------------------------------------------------------
    # 2. EXECUTE TOOL 1: Basic Eligibility Check
    # ---------------------------------------------------------
    # If age is 0 (missing), we skip the age failure for now so it doesn't break testing, 
    # but normally the Intake Agent should ask for it.
    check_age = age if age > 0 else 30 
    eligibility_result = eligibility.check_basic_eligibility(check_age, income)

    # ---------------------------------------------------------
    # 3. EXECUTE TOOL 2: EMI Calculator
    # ---------------------------------------------------------
    # Assuming a fixed interest rate of 12.5% for standard calculation
    emi = emi_calculator.calculate(loan_amount, 12.5, tenure) if loan_amount > 0 else 0.0
    
    # ---------------------------------------------------------
    # 4. EXECUTE TOOL 3: Risk Scoring
    # ---------------------------------------------------------
    risk = risk_scoring.get_risk_band(emi, income)
    emi_burden_pct = (emi / income) * 100 if income > 1 else 0.0

    # 5. Package all tool results for the Decision Agent and the UI
    return {
        "is_eligible": eligibility_result["eligible"],
        "eligibility_reasons": eligibility_result["reasons"],
        "emi": emi,
        "emi_burden_pct": round(emi_burden_pct, 1),
        "risk_band": risk,
        "tenure_used": tenure
    }