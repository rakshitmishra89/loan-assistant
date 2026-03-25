"""
perf/cache.py
=============
LangChain LLM caching layer for the Loan Assistant.

This project uses ChatOllama (Mistral, local) as the LLM.
LangChain's set_llm_cache() hooks into ChatOllama automatically —
so once activated, repeated identical prompts are served from cache
without hitting the model again.

Two cache backends:
  1. InMemoryCache  – dev/testing  (lost on restart)
  2. SQLiteCache    – production   (persists on disk)

TTL Strategy
------------
LangChain's built-in caches have no TTL, so we add a thin TTLCache
wrapper for time-based expiry. This matters because:
  - Bank interest rates and policy rules change periodically.
  - Personal/dynamic prompts (income, credit score) must NEVER be cached.
  - Generic FAQ answers are safe to cache for a few hours.

Safe Caching Rules
------------------
  ✅ CACHE   → generic policy, FAQ, interest rate questions
  ❌ NO CACHE → any prompt with PII or personal financial data

PII keywords are kept in sync with guardrails/guardrails.py
to ensure consistent detection across the pipeline.
"""

import hashlib
import sqlite3
import time
import threading
from typing import Any, Optional

import langchain
from langchain_community.cache import InMemoryCache, SQLiteCache

def set_llm_cache(cache):
    langchain.llm_cache = cache

def set_llm_cache(cache):
    langchain.llm_cache = cache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TTL_SECONDS: int = 4 * 60 * 60           # 4 hours default; tune as needed
SQLITE_DB_PATH: str = "perf/llm_cache.db"

# ---------------------------------------------------------------------------
# PII / Dynamic-data keywords
# Must stay in sync with guardrails/guardrails.py PII_PATTERNS keys.
# Prompts matching ANY of these bypass the cache entirely.
# ---------------------------------------------------------------------------
_NO_CACHE_KEYWORDS = [
    # Personal financial data (extracted by intake_agent)
    "income", "salary", "loan amount", "credit score", "cibil",
    "tenure", "age", "emi",
    # PII identifiers (already detected + redacted by guardrails,
    # but we still skip caching as an extra safety layer)
    "aadhaar", "pan", "phone", "email", "dob", "date of birth",
    "passport", "voter", "ifsc", "bank account",
    # Applicant-specific terms
    "my name", "applicant", "co-applicant", "guarantor",
    "property value", "collateral", "employment",
]


# ---------------------------------------------------------------------------
# Thread-safe TTL Cache
# ---------------------------------------------------------------------------

class TTLCache:
    """
    In-memory key-value store with per-entry TTL expiry.
    Thread-safe via a single Lock.

    Used by the orchestrator for sub-second repeated query caching
    within the same server process (e.g. same FAQ asked by many users).
    """

    def __init__(self, ttl: int = TTL_SECONDS):
        self.ttl = ttl
        # store: key -> (value, inserted_timestamp)
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, ts = entry
            if (time.time() - ts) < self.ttl:
                return value
            del self._store[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """Store a value with the current timestamp."""
        with self._lock:
            self._store[key] = (value, time.time())

    def delete(self, key: str) -> bool:
        """Manually evict a key. Returns True if key existed."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Flush the entire cache."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        """Return live stats — useful for the latency notebook."""
        with self._lock:
            now = time.time()
            total = len(self._store)
            expired = sum(
                1 for _, ts in self._store.values()
                if (now - ts) >= self.ttl
            )
            return {
                "total_entries": total,
                "live_entries": total - expired,
                "expired_entries": expired,
                "ttl_seconds": self.ttl,
            }


# ---------------------------------------------------------------------------
# Module-level cache instances (singletons shared across FastAPI process)
# ---------------------------------------------------------------------------

llm_cache = TTLCache(ttl=300)        # 5 min  — for LLM replies
retrieval_cache = TTLCache(ttl=600)  # 10 min — for RAG chunk results


# ---------------------------------------------------------------------------
# Safe-caching guard
# ---------------------------------------------------------------------------

def is_cacheable(prompt: str) -> bool:
    """
    Returns True only if the prompt is safe to cache.

    Rejects any prompt that contains personal financial data or PII
    since those responses are unique per applicant and must not be
    served to other users.

    Examples
    --------
    >>> is_cacheable("What is the home loan interest rate?")
    True
    >>> is_cacheable("My income is 80000. Am I eligible?")
    False
    >>> is_cacheable("My credit score is 720")
    False
    """
    lower = prompt.lower()
    return not any(keyword in lower for keyword in _NO_CACHE_KEYWORDS)


def make_cache_key(prompt: str, model: str = "mistral") -> str:
    """
    Deterministic MD5 cache key from model name + prompt.
    Model name included so switching models invalidates old entries.
    """
    raw = f"{model}::{prompt.strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LangChain cache setup — call ONE of these at app startup (in main.py)
# ---------------------------------------------------------------------------

def setup_inmemory_cache() -> tuple:
    """
    Activate LangChain's InMemoryCache for ChatOllama (Mistral).
    Use this in development and unit tests.

    Returns (llm_cache, retrieval_cache) for use in the orchestrator.

    Usage in main.py:
        from perf.cache import setup_inmemory_cache
        llm_c, rag_c = setup_inmemory_cache()
    """
    set_llm_cache(InMemoryCache())
    print("[perf/cache] ✅ LangChain InMemoryCache activated (dev mode).")
    return llm_cache, retrieval_cache


def setup_sqlite_cache(db_path: str = SQLITE_DB_PATH) -> tuple:
    """
    Activate LangChain's SQLiteCache (persistent) for ChatOllama (Mistral).
    Use this in staging / production.

    Returns (llm_cache, retrieval_cache) for use in the orchestrator.

    Usage in main.py:
        from perf.cache import setup_sqlite_cache
        llm_c, rag_c = setup_sqlite_cache()
    """
    set_llm_cache(SQLiteCache(database_path=db_path))
    print(f"[perf/cache] ✅ LangChain SQLiteCache activated → {db_path}")
    return llm_cache, retrieval_cache


# ---------------------------------------------------------------------------
# Cached LLM call wrapper
# Use this in decision_agent.py instead of chain.invoke() directly
# ---------------------------------------------------------------------------

async def cached_llm_call(
    prompt: str,
    llm_coroutine,              # async callable: () -> str
    model: str = "mistral",
    cache: TTLCache = llm_cache,
) -> tuple:
    """
    Wraps an async ChatOllama call with TTL caching.

    Parameters
    ----------
    prompt        : Full prompt string sent to the LLM.
    llm_coroutine : Async callable with no args that returns a str response.
    model         : LLM model name used in key generation (default: mistral).
    cache         : TTLCache instance (default: module-level llm_cache).

    Returns
    -------
    (response_text, cache_hit)
      cache_hit=True  → served from cache, LLM was NOT called
      cache_hit=False → LLM was called, result stored in cache

    How to use in decision_agent.py:
        from perf.cache import cached_llm_call
        reply, hit = await cached_llm_call(
            prompt=full_prompt,
            llm_coroutine=lambda: chain.ainvoke({...}).content
        )
    """
    if not is_cacheable(prompt):
        # Personal/dynamic data — always call LLM fresh, never cache
        response = await llm_coroutine()
        return response, False

    key = make_cache_key(prompt, model)
    cached_response = cache.get(key)
    if cached_response is not None:
        return cached_response, True          # ✅ Cache HIT

    # Cache MISS — call the real LLM and store result
    response = await llm_coroutine()
    cache.set(key, response)
    return response, False


# ---------------------------------------------------------------------------
# Cached RAG retrieval wrapper
# Use this in retrieval_agent.py instead of rag_adapter.retrieve() directly
# ---------------------------------------------------------------------------

def cached_retrieval(
    query: str,
    retrieval_fn,               # callable: (query: str) -> dict
    cache: TTLCache = retrieval_cache,
) -> tuple:
    """
    Wraps the RAG retrieval call (ChromaDB lookup) with TTL caching.

    Generic policy queries return identical chunks every time —
    caching avoids repeated vector similarity searches.

    Parameters
    ----------
    query        : User's message / search query.
    retrieval_fn : Callable that takes a query string and returns:
                   {"used_rag": bool, "chunks": list}
                   (This is retrieval_agent.process() in the project)
    cache        : TTLCache instance.

    Returns
    -------
    (rag_result_dict, cache_hit)

    How to use in retrieval_agent.py:
        from perf.cache import cached_retrieval
        rag_data, hit = cached_retrieval(safe_message, process)
    """
    if not is_cacheable(query):
        return retrieval_fn(query), False

    key = make_cache_key(query, model="rag")
    cached_result = cache.get(key)
    if cached_result is not None:
        return cached_result, True            # ✅ Cache HIT

    result = retrieval_fn(query)
    cache.set(key, result)
    return result, False


# ---------------------------------------------------------------------------
# SQLite TTL purge utility — run at startup to keep DB lean
# ---------------------------------------------------------------------------

def purge_expired_sqlite_entries(
    db_path: str = SQLITE_DB_PATH,
    ttl_seconds: int = TTL_SECONDS,
) -> int:
    """
    Deletes rows from the LangChain SQLite cache older than ttl_seconds.
    Returns the number of rows deleted.

    LangChain's SQLiteCache table is 'full_llm_cache'. We maintain a
    shadow 'llm_cache_timestamps' table since LangChain stores no timestamps.
    """
    cutoff = time.time() - ttl_seconds
    deleted = 0
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache_timestamps (
                prompt_hash TEXT PRIMARY KEY,
                inserted_at REAL
            )
        """)
        cur.execute(
            "SELECT prompt_hash FROM llm_cache_timestamps WHERE inserted_at < ?",
            (cutoff,),
        )
        stale = [row[0] for row in cur.fetchall()]
        if stale:
            ph = ",".join("?" * len(stale))
            cur.execute(f"DELETE FROM full_llm_cache WHERE prompt_hash IN ({ph})", stale)
            cur.execute(
                f"DELETE FROM llm_cache_timestamps WHERE prompt_hash IN ({ph})", stale
            )
            deleted = len(stale)
        con.commit()
        con.close()
    except Exception as e:
        print(f"[perf/cache] ⚠️  purge error: {e}")
    return deleted


# ---------------------------------------------------------------------------
# Self-test  →  python -m perf.cache
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== perf/cache.py — Self Test ===\n")

    # 1. is_cacheable
    assert is_cacheable("What is the home loan interest rate?") is True
    assert is_cacheable("My income is 50000, am I eligible?") is False
    assert is_cacheable("My credit score is 720") is False
    assert is_cacheable("What documents are needed for a car loan?") is True
    print("✅ is_cacheable() — all assertions passed")

    # 2. TTLCache get/set
    c = TTLCache(ttl=2)
    c.set("hello", "world")
    assert c.get("hello") == "world"
    print("✅ TTLCache.get/set — OK")

    # 3. TTLCache expiry
    time.sleep(3)
    assert c.get("hello") is None, "Should have expired"
    print("✅ TTLCache TTL expiry — OK")

    # 4. Stats
    c2 = TTLCache(ttl=60)
    c2.set("a", 1)
    c2.set("b", 2)
    stats = c2.stats()
    assert stats["live_entries"] == 2
    print(f"✅ TTLCache.stats() — {stats}")

    # 5. LangChain setup
    setup_inmemory_cache()
    print("✅ setup_inmemory_cache() — OK")

    print("\n🎉 All tests passed!")
