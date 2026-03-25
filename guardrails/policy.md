# Guardrails Policy Document
**Project:** Loan Approval & Credit Risk Assistant  
**Member:** 4 — Guardrails Engineer  

---

## 1. What Guardrails Does

Guardrails is a **safety agent** that sits at two points in the pipeline:

- **Step 2 — Input Guard:** Checks user message BEFORE it reaches the LLM
- **Step 7 — Output Guard:** Checks LLM response BEFORE it reaches the user

---

## 2. Agent Decision Flow

```
Message comes in
      ↓
Agent checks: self_harm?  → BLOCK + helpline numbers
Agent checks: hate_speech?→ BLOCK + guideline warning  
Agent checks: abuse?      → BLOCK + respectful warning
Agent checks: profanity?  → BLOCK + language warning
Agent checks: PII?        → REDACT + allow message
Agent checks: clean?      → PASS through unchanged
```

---

## 3. Categories & Actions

| Category | Examples | Action | Allowed? |
|---|---|---|---|
| `clean` | Normal loan questions | PASS | ✅ Yes |
| `pii` | Aadhaar, PAN, phone | REDACT | ✅ Yes (cleaned) |
| `profanity` | Swear words, Hindi abuses | BLOCK | ❌ No |
| `hate_speech` | Racial/religious slurs | BLOCK | ❌ No |
| `abuse` | Threats, personal attacks | BLOCK | ❌ No |
| `self_harm` | Suicide/self-harm mentions | BLOCK | ❌ No |

---

## 4. Safe Response Templates

### Profanity
```
⚠️ Your message contains inappropriate language.
Please keep the conversation professional so I can assist you
with your loan application.
```

### Hate Speech
```
🚫 Your message contains content that violates our guidelines.
We provide equal, respectful service to all applicants.
Please rephrase your message.
```

### Abuse
```
⚠️ Your message contains abusive content.
Our team is here to help you. Please communicate respectfully
so we can process your loan application.
```

### Self Harm (HIGHEST PRIORITY)
```
💙 It sounds like you are going through a very difficult time.
Your wellbeing matters more than any loan.
Please reach out for help:
  • iCall (India): 9152987821
  • Vandrevala Foundation: 1860-2662-345
We are here for you when you are ready to continue.
```

### PII Detected (informational only)
```
🔒 Sensitive personal information was detected and hidden
for your security. Please use our secure document upload
instead of sharing IDs in chat.
```

---

## 5. PII Patterns & Redaction

| PII Type | Pattern Description | Replaced With |
|---|---|---|
| Aadhaar | 12-digit number starting 2-9 | `[AADHAAR REDACTED]` |
| PAN Card | Format: AAAAA9999A | `[PAN REDACTED]` |
| Phone | 10-digit Indian mobile | `[PHONE REDACTED]` |
| Email | Standard email format | `[EMAIL REDACTED]` |
| Credit Card | 13-16 digit number | `[CARD REDACTED]` |
| Bank Account | 9-18 digit number | `[ACCOUNT REDACTED]` |
| IFSC Code | Format: AAAA0AAAAAA | `[IFSC REDACTED]` |
| Date of Birth | DD/MM/YYYY format | `[DOB REDACTED]` |
| Passport | Indian passport format | `[PASSPORT REDACTED]` |
| Voter ID | Format: AAA9999999 | `[VOTERID REDACTED]` |

### Redaction Rules
- PII is replaced — never stored
- Redaction happens before LLM sees the text
- Multiple PII types in one message — all get redacted
- Message still allowed after redaction

---

## 6. How Member 1 Uses This

```python
from guardrails.guardrails import (
    moderate_input,
    moderate_output,
    redact_pii,
    get_safe_response
)

# In /chat endpoint:
input_result = moderate_input(user_message)
if not input_result["allowed"]:
    return get_safe_response(input_result["category"])

# Send input_result["redacted_text"] to LLM
llm_response = llm.invoke(input_result["redacted_text"])

# Check output
output_result = moderate_output(llm_response)
return output_result["safe_text"]
```

---

## 7. Priority Order (Why It Matters)

Self-harm is checked FIRST because human safety > everything else.
PII is checked LAST because it's not harmful — just needs cleaning.

```
1st → self_harm   (life safety)
2nd → hate_speech (community safety)
3rd → abuse       (user safety)
4th → profanity   (professional standards)
5th → pii         (data privacy)
6th → clean       (normal flow)
```
