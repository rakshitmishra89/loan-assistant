from tools import emi_calculator, risk_scoring, eligibility

def run_all(state_entities: dict) -> dict:
    """
    Acts as the bridge between memory and tools with enhanced error handling. 
    Sanitizes data and ensures the principal is passed for UI visualization.
    """
    # 1. Safely extract and sanitize values from memory
    try:
        loan_amount = float(state_entities.get("loan_amount") or 0.0)
        income = float(state_entities.get("income_monthly") or 0.0)
        tenure = int(state_entities.get("tenure_months") or 36)
        age = int(state_entities.get("age") or 0)
        credit_score = int(state_entities.get("credit_score") or 0)
    except (ValueError, TypeError):
        # Fallback for invalid numeric data types
        return {
            "is_eligible": False, 
            "eligibility_reasons": ["Invalid data format detected in application."], 
            "emi": 0.0, 
            "risk_band": "HIGH"
        }

    # ---------------------------------------------------------
    # 2. EXECUTE TOOL 1: Basic Eligibility Check
    # ---------------------------------------------------------
    # Default to 30 if age is unknown to allow policy information retrieval 
    # while the Intake Agent gathers more data.
    check_age = age if age > 0 else 30 
    eligibility_result = eligibility.check_basic_eligibility(check_age, income)

    # ---------------------------------------------------------
    # 3. EXECUTE TOOL 2: EMI Calculator
    # ---------------------------------------------------------
    # Standard reducing balance calculation at 12.5%
    emi = emi_calculator.calculate(loan_amount, 12.5, tenure) if loan_amount > 0 else 0.0
    
    # ---------------------------------------------------------
    # 4. EXECUTE TOOL 3: Risk Scoring
    # ---------------------------------------------------------
    # Pass the credit score for policy-based rejection
    risk = risk_scoring.get_risk_band(emi, income, credit_score)
    
    # Calculate Burden safely (Shielded from ZeroDivision)
    emi_burden_pct = (emi / income) * 100 if income > 0 else 0.0

    # 5. Package results for the Decision Agent and UI Visualizations
    return {
        "is_eligible": eligibility_result["eligible"],
        "eligibility_reasons": eligibility_result["reasons"],
        "emi": emi,
        "emi_burden_pct": round(emi_burden_pct, 1),
        "risk_band": risk,
        "tenure_used": tenure,
        "principal": loan_amount
    }