# perf/error_tests.md
# Error & Edge-Case Test Plan — Loan Assistant

This document covers manual and automated error tests for the Loan Assistant API.
Each test includes the **input**, **expected behaviour**, and **actual result** column
(fill in during testing).

---

## 1. Missing / Empty Input Tests

### 1.1 Empty message field

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": "abc123", "message": "", "metadata": {}}` |
| **Expected** | `422 Unprocessable Entity` — validation error on `message` field |
| **Actual**   | _(fill during test)_ |

---

### 1.2 Missing `message` field entirely

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": "abc123", "metadata": {}}` |
| **Expected** | `422 Unprocessable Entity` — `message` is required |
| **Actual**   | _(fill during test)_ |

---

### 1.3 Missing `session_id`

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"message": "What is the interest rate?", "metadata": {}}` |
| **Expected** | `422 Unprocessable Entity` — `session_id` is required |
| **Actual**   | _(fill during test)_ |

---

### 1.4 Null values

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": null, "message": null, "metadata": null}` |
| **Expected** | `422` — nulls rejected by Pydantic schema |
| **Actual**   | _(fill during test)_ |

---

### 1.5 Empty RAG query

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /rag/query` |
| **Payload**  | `{"query": "", "k": 3}` |
| **Expected** | `422` or graceful empty-result response |
| **Actual**   | _(fill during test)_ |

---

## 2. Invalid Data Tests

### 2.1 Extremely long message (token overflow)

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": "s1", "message": "A" * 50000, "metadata": {}}` |
| **Expected** | `400` or `422` — message exceeds max length; or graceful truncation with warning |
| **Actual**   | _(fill during test)_ |

---

### 2.2 Invalid `k` in RAG query (negative or zero)

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /rag/query` |
| **Payload**  | `{"query": "home loan rate", "k": -1}` |
| **Expected** | `422` — `k` must be a positive integer |
| **Actual**   | _(fill during test)_ |

---

### 2.3 Wrong data types

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": 12345, "message": ["list", "instead", "of", "string"], "metadata": "bad"}` |
| **Expected** | `422` — Pydantic type coercion errors |
| **Actual**   | _(fill during test)_ |

---

### 2.4 SQL/prompt injection attempt

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": "s1", "message": "'; DROP TABLE loans; --", "metadata": {}}` |
| **Expected** | Safe — treated as plain text, guardrails flag as off-topic |
| **Actual**   | _(fill during test)_ |

---

### 2.5 XSS payload in message

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": "s1", "message": "<script>alert('xss')</script>", "metadata": {}}` |
| **Expected** | Safe — returned as escaped text, not executed |
| **Actual**   | _(fill during test)_ |

---

### 2.6 Unicode / emoji / non-ASCII message

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": "s1", "message": "गृह ऋण ब्याज दर क्या है? 🏠💰", "metadata": {}}` |
| **Expected** | `200` — handled gracefully, LLM responds (possibly in English) |
| **Actual**   | _(fill during test)_ |

---

### 2.7 Completely off-topic message (guardrails test)

| Field        | Value |
|--------------|-------|
| **Endpoint** | `POST /chat` |
| **Payload**  | `{"session_id": "s1", "message": "Write me a poem about the moon.", "metadata": {}}` |
| **Expected** | `200` — guardrails block with polite redirect message |
| **Actual**   | _(fill during test)_ |

---

## 3. Bad File Upload Tests (Frontend / Streamlit)

### 3.1 Unsupported file type

| Field        | Value |
|--------------|-------|
| **Trigger**  | Upload `.exe` or `.zip` via Streamlit file uploader |
| **Expected** | File uploader rejects the file (only `pdf`, `txt` allowed) |
| **Actual**   | _(fill during test)_ |

---

### 3.2 Empty file (0 bytes)

| Field        | Value |
|--------------|-------|
| **Trigger**  | Upload a 0-byte `.txt` file |
| **Expected** | Graceful error message: "File appears to be empty" |
| **Actual**   | _(fill during test)_ |

---

### 3.3 Very large file (> 10 MB)

| Field        | Value |
|--------------|-------|
| **Trigger**  | Upload a PDF > 10 MB |
| **Expected** | Warning message or rejection with file-size limit notice |
| **Actual**   | _(fill during test)_ |

---

### 3.4 Corrupted PDF

| Field        | Value |
|--------------|-------|
| **Trigger**  | Upload a `.pdf` file with corrupted/truncated bytes |
| **Expected** | Error caught during processing; friendly error message shown |
| **Actual**   | _(fill during test)_ |

---

### 3.5 File with no readable text (scanned image PDF)

| Field        | Value |
|--------------|-------|
| **Trigger**  | Upload a scanned PDF with no text layer |
| **Expected** | Graceful warning: "Could not extract text from this document" |
| **Actual**   | _(fill during test)_ |

---

## 4. Backend Unavailability Tests

### 4.1 FastAPI backend not running

| Field        | Value |
|--------------|-------|
| **Trigger**  | Start Streamlit with FastAPI offline |
| **Expected** | Streamlit shows: `🚨 Cannot connect to FastAPI backend. Did you run uvicorn...?` |
| **Actual**   | _(fill during test)_ |
| **Notes**    | Already handled in `frontend/app.py` with `ConnectionError` catch |

---

### 4.2 Slow backend response (timeout)

| Field        | Value |
|--------------|-------|
| **Trigger**  | Simulate 35-second backend delay |
| **Expected** | Streamlit shows timeout error; stress_test records `TIMEOUT` status |
| **Actual**   | _(fill during test)_ |

---

## 5. How to Run These Tests (Automated)

You can run HTTP-based tests using `pytest` + `httpx`:

```bash
pip install pytest httpx
pytest perf/test_errors.py -v
```

### Sample `perf/test_errors.py` snippet

```python
import pytest
import httpx

BASE = "http://localhost:8000"

@pytest.mark.asyncio
async def test_empty_message():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/chat", json={
            "session_id": "test", "message": "", "metadata": {}
        })
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_missing_session_id():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/chat", json={
            "message": "What is the interest rate?", "metadata": {}
        })
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_rag_negative_k():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/rag/query", json={
            "query": "home loan", "k": -1
        })
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_very_long_message():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/chat", json={
            "session_id": "test", "message": "A" * 50000, "metadata": {}
        })
    assert r.status_code in (400, 422)
```

---

## 6. Test Results Summary Table

| # | Test Case | Expected | Pass/Fail | Notes |
|---|-----------|----------|-----------|-------|
| 1.1 | Empty message | 422 | | |
| 1.2 | Missing message field | 422 | | |
| 1.3 | Missing session_id | 422 | | |
| 1.4 | Null values | 422 | | |
| 1.5 | Empty RAG query | 422/empty | | |
| 2.1 | 50k char message | 400/422 | | |
| 2.2 | Negative k | 422 | | |
| 2.3 | Wrong data types | 422 | | |
| 2.4 | SQL injection | Safe/200 | | |
| 2.5 | XSS payload | Safe/200 | | |
| 2.6 | Unicode/emoji | 200 | | |
| 2.7 | Off-topic message | 200 + guardrail | | |
| 3.1 | Unsupported file type | Rejected in UI | | |
| 3.2 | Empty file | Graceful error | | |
| 3.3 | File > 10 MB | Warning/reject | | |
| 3.4 | Corrupted PDF | Graceful error | | |
| 3.5 | Scanned image PDF | Warning | | |
| 4.1 | Backend offline | ConnectionError msg | | |
| 4.2 | Slow backend | Timeout | | |
