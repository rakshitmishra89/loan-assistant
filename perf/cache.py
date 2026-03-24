import time
import hashlib
import threading
from typing import Any, Dict, Tuple

# -----------------------------
# Thread-safe TTL Cache
# -----------------------------
class TTLCache:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.store: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.store:
                value, ts = self.store[key]
                if (time.time() - ts) < self.ttl:
                    return value
                else:
                    del self.store[key]
        return None

    def set(self, key: str, value: Any):
        with self.lock:
            self.store[key] = (value, time.time())


# -----------------------------
# Cache Instances
# -----------------------------
llm_cache = TTLCache(ttl=300) # 5 min
retriever_cache = TTLCache(ttl=600) # 10 min


# -----------------------------
# Helper
# -----------------------------
def hash_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# -----------------------------
# Cached LLM Call
# -----------------------------
def cached_llm_call(llm, prompt: str):
    # Avoid caching sensitive data
    if any(word in prompt.lower() for word in ["aadhaar", "phone", "salary"]):
        return llm.invoke(prompt)

    key = hash_key(prompt)
    cached = llm_cache.get(key)

    if cached:
        print("⚡ LLM Cache HIT")
        return cached

    print("🐢 LLM Cache MISS")
    response = llm.invoke(prompt)
    llm_cache.set(key, response)
    return response


# -----------------------------
# Cached Retriever Call
# -----------------------------
def cached_retrieval(retriever, query: str):
    key = hash_key(query)
    cached = retriever_cache.get(key)

    if cached:
        print("⚡ Retriever Cache HIT")
        return cached

    print("🐢 Retriever Cache MISS")
    docs = retriever.get_relevant_documents(query)
    retriever_cache.set(key, docs)
    return docs
