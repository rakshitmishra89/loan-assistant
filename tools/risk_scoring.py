def get_risk_band(emi: float, monthly_income: float, credit_score: int) -> str:
    """Calculates Debt-to-Income (DTI) and applies strict CIBIL checks with defensive math."""
    
    # 1. Strict CIBIL Check
    # Automatic rejection if score is below 650 as per Master Policy
    if 0 < credit_score < 650:
        return "HIGH" 
        
    # 2. Income Safety Check (Prevent Division by Zero)
    if not monthly_income or monthly_income <= 0:
        return "HIGH" # Cannot determine risk without verifiable income
        
    # 3. EMI Safety Check
    if emi < 0:
        return "HIGH" # Data integrity issue
        
    # 4. DTI (Debt-to-Income) Calculation
    emi_burden_pct = (emi / monthly_income) * 100
    
    # Thresholds as per automated underwriting rules
    if emi_burden_pct > 45:
        return "HIGH"
    elif emi_burden_pct > 30:
        return "MEDIUM"
    else:
        return "LOW"