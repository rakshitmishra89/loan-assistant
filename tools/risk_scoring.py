def get_risk_band(emi: float, monthly_income: float) -> str:
    """Calculates Debt-to-Income (DTI) and assigns a risk band."""
    if not monthly_income or monthly_income <= 0:
        return "HIGH"
        
    emi_burden_pct = (emi / monthly_income) * 100
    
    if emi_burden_pct > 45:
        return "HIGH"
    elif emi_burden_pct > 30:
        return "MEDIUM"
    else:
        return "LOW"