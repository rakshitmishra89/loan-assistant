# perf/__init__.py
"""
Performance utilities for the Loan Assistant.

- cache.py: LLM and RAG caching with TTL
- stress_test.py: Load testing utility
- error_tests.md: Error test documentation
"""

from perf.cache import (
    TTLCache,
    llm_cache,
    retrieval_cache,
    is_cacheable,
    make_cache_key,
    cached_llm_call,
    cached_retrieval,
    setup_inmemory_cache,
    setup_sqlite_cache,
)

__all__ = [
    "TTLCache",
    "llm_cache",
    "retrieval_cache", 
    "is_cacheable",
    "make_cache_key",
    "cached_llm_call",
    "cached_retrieval",
    "setup_inmemory_cache",
    "setup_sqlite_cache",
]
