def check_basic_eligibility(age: int, income_monthly: float) -> dict:
    """Checks basic bank policy rules: Age 21-60, Min Income $3500"""
    is_eligible = True
    reasons = []

    if age and (age < 21 or age > 60):
        is_eligible = False
        reasons.append("Applicant age must be between 21 and 60.")
        
    if income_monthly and income_monthly < 3500:
        is_eligible = False
        reasons.append("Minimum monthly income must be $3,500.")
        
    return {"eligible": is_eligible, "reasons": reasons}