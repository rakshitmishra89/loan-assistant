def calculate(principal: float, rate_annual: float, months: int) -> float:
    """Standard reducing balance EMI formula."""
    if not principal or not rate_annual or not months: 
        return 0.0
    r = (rate_annual / 12) / 100
    emi = principal * r * ((1 + r)**months) / (((1 + r)**months) - 1)
    return round(emi, 2)