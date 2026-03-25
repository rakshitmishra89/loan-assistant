<<<<<<< HEAD
"""
perf/stress_test.py
===================
Parallel stress-test for the Loan Assistant FastAPI backend.

What it does
------------
  1. Fires N concurrent POST /chat requests using asyncio + httpx.
  2. Measures per-request latency and overall throughput.
  3. Tracks success / error rates.
  4. Prints a clean summary table and saves results to
     perf/stress_results.json for use in latency.ipynb.

Usage
-----
    # Default: 20 concurrent users, 50 total requests
    python -m perf.stress_test

    # Custom load
    python -m perf.stress_test --users 50 --requests 200 --url http://localhost:8000

Requirements
------------
    pip install httpx rich
"""

import asyncio
import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False

# ---------------------------------------------------------------------------
# Test payloads  — realistic loan-assistant queries
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    "What is the current home loan interest rate?",
    "What documents are required for a personal loan?",
    "How is EMI calculated?",
    "What is the maximum tenure for a car loan?",
    "What is the minimum CIBIL score needed for approval?",
    "Explain the difference between fixed and floating interest rates.",
    "What are the prepayment charges on a home loan?",
    "How long does the loan approval process take?",
    "Can I apply for a loan if I am self-employed?",
    "What is a loan-to-value ratio?",
]

RESULTS_PATH = Path("perf/stress_results.json")

# ---------------------------------------------------------------------------
# Single request coroutine
# ---------------------------------------------------------------------------

async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
) -> dict[str, Any]:
    """Fire one POST /chat and return timing + result metadata."""
    session_id = str(uuid.uuid4())
    message = SAMPLE_MESSAGES[index % len(SAMPLE_MESSAGES)]
    payload = {
        "session_id": session_id,
        "message": message,
        "metadata": {"channel": "stress_test", "request_index": index},
    }

    start = time.perf_counter()
    result = {
        "index": index,
        "message": message,
        "session_id": session_id,
        "status": None,
        "latency_ms": None,
        "backend_llm_ms": None,
        "backend_e2e_ms": None,
        "cache_hit": False,
        "error": None,
    }

    try:
        response = await client.post(f"{base_url}/chat", json=payload, timeout=30.0)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        result["status"] = response.status_code
        result["latency_ms"] = elapsed_ms

        if response.status_code == 200:
            data = response.json()
            latency_info = data.get("latency_ms", {})
            result["backend_llm_ms"] = latency_info.get("llm")
            result["backend_e2e_ms"] = latency_info.get("end_to_end")
            result["cache_hit"] = data.get("cache_hit", False)
        else:
            result["error"] = response.text

    except httpx.TimeoutException:
        result["status"] = "TIMEOUT"
        result["error"] = "Request timed out after 30s"
        result["latency_ms"] = 30_000
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

    return result


# ---------------------------------------------------------------------------
# Concurrent batch runner
# ---------------------------------------------------------------------------

async def run_stress_test(
    base_url: str,
    total_requests: int,
    concurrent_users: int,
) -> list[dict]:
    """
    Run `total_requests` requests with at most `concurrent_users` in-flight
    at any one time (semaphore-controlled).
    """
    semaphore = asyncio.Semaphore(concurrent_users)
    results: list[dict] = []

    async def bounded_request(client, idx):
        async with semaphore:
            return await send_request(client, base_url, idx)

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client, i) for i in range(total_requests)]
        print(f"\n🚀 Firing {total_requests} requests "
              f"({concurrent_users} concurrent) at {base_url} ...\n")
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_elapsed = round((time.perf_counter() - wall_start) * 1000, 2)

    return results, wall_elapsed


# ---------------------------------------------------------------------------
# Summary & reporting
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict], wall_ms: float) -> dict:
    total = len(results)
    successes = [r for r in results if r["status"] == 200]
    failures  = [r for r in results if r["status"] != 200]
    cache_hits = [r for r in successes if r["cache_hit"]]

    latencies = [r["latency_ms"] for r in successes if r["latency_ms"] is not None]
    llm_times = [r["backend_llm_ms"] for r in successes if r["backend_llm_ms"]]
    e2e_times = [r["backend_e2e_ms"] for r in successes if r["backend_e2e_ms"]]

    def stats(lst):
        if not lst:
            return {}
        lst_sorted = sorted(lst)
        n = len(lst_sorted)
        return {
            "min":  round(min(lst_sorted), 2),
            "max":  round(max(lst_sorted), 2),
            "avg":  round(sum(lst_sorted) / n, 2),
            "p50":  round(lst_sorted[int(n * 0.50)], 2),
            "p90":  round(lst_sorted[int(n * 0.90)], 2),
            "p95":  round(lst_sorted[int(n * 0.95)], 2),
            "p99":  round(lst_sorted[min(int(n * 0.99), n - 1)], 2),
        }

    return {
        "run_at": datetime.utcnow().isoformat(),
        "total_requests": total,
        "successful": len(successes),
        "failed": len(failures),
        "cache_hits": len(cache_hits),
        "cache_hit_rate_pct": round(len(cache_hits) / total * 100, 1) if total else 0,
        "wall_time_ms": wall_ms,
        "throughput_rps": round(total / (wall_ms / 1000), 2),
        "client_latency_ms": stats(latencies),
        "backend_llm_ms": stats(llm_times),
        "backend_e2e_ms": stats(e2e_times),
        "errors": [
            {"index": r["index"], "status": r["status"], "error": r["error"]}
            for r in failures
        ],
    }


def print_summary(summary: dict) -> None:
    if _RICH:
        _print_rich(summary)
    else:
        _print_plain(summary)


def _print_rich(s: dict) -> None:
    console = Console()
    console.print("\n[bold cyan]═══ Stress Test Summary ═══[/bold cyan]\n")

    # Overview table
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Total requests",  str(s["total_requests"]))
    t.add_row("Successful",      f"[green]{s['successful']}[/green]")
    t.add_row("Failed",          f"[red]{s['failed']}[/red]" if s["failed"] else "0")
    t.add_row("Cache hits",      f"{s['cache_hits']} ({s['cache_hit_rate_pct']}%)")
    t.add_row("Wall time",       f"{s['wall_time_ms']} ms")
    t.add_row("Throughput",      f"{s['throughput_rps']} req/s")
    console.print(t)

    # Latency table
    console.print("\n[bold cyan]Latency breakdown (ms)[/bold cyan]\n")
    lt = Table(show_header=True, header_style="bold magenta")
    lt.add_column("Metric")
    for col in ["min", "avg", "p50", "p90", "p95", "p99", "max"]:
        lt.add_column(col, justify="right")

    def row(label, d):
        if d:
            lt.add_row(label, *[str(d.get(c, "-")) for c in ["min","avg","p50","p90","p95","p99","max"]])

    row("Client latency",   s["client_latency_ms"])
    row("Backend LLM",      s["backend_llm_ms"])
    row("Backend E2E",      s["backend_e2e_ms"])
    console.print(lt)

    if s["errors"]:
        console.print(f"\n[red]⚠ {len(s['errors'])} error(s):[/red]")
        for e in s["errors"][:5]:
            console.print(f"  #{e['index']} → {e['status']}: {e['error']}")


def _print_plain(s: dict) -> None:
    print("\n=== Stress Test Summary ===")
    print(f"Total:       {s['total_requests']}")
    print(f"Successful:  {s['successful']}")
    print(f"Failed:      {s['failed']}")
    print(f"Cache hits:  {s['cache_hits']} ({s['cache_hit_rate_pct']}%)")
    print(f"Wall time:   {s['wall_time_ms']} ms")
    print(f"Throughput:  {s['throughput_rps']} req/s")
    if s["client_latency_ms"]:
        lat = s["client_latency_ms"]
        print(f"\nClient latency (ms): avg={lat['avg']} p90={lat['p90']} p99={lat['p99']}")
    if s["errors"]:
        print(f"\n⚠  {len(s['errors'])} error(s):")
        for e in s["errors"][:5]:
            print(f"  #{e['index']} → {e['status']}: {e['error']}")


def save_results(results: list[dict], summary: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "requests": results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Results saved → {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stress test the Loan Assistant FastAPI backend."
    )
    parser.add_argument("--url",      default="http://localhost:8000", help="Base URL of the backend")
    parser.add_argument("--users",    type=int, default=20,  help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=50,  help="Total number of requests")
    parser.add_argument("--save",     action="store_true", default=True, help="Save results to JSON")
    return parser.parse_args()


async def main():
    args = parse_args()
    results, wall_ms = await run_stress_test(
        base_url=args.url,
        total_requests=args.requests,
        concurrent_users=args.users,
    )
    summary = compute_summary(results, wall_ms)
    print_summary(summary)
    if args.save:
        save_results(results, summary)


if __name__ == "__main__":
    asyncio.run(main())
=======
import requests
import threading
import time

URL = "http://127.0.0.1:8000/chat"

def send_request(i):
    try:
        res = requests.post(URL, json={"message": "I want a loan"})
        print(f"[{i}] {res.status_code}")
    except Exception as e:
        print(f"[{i}] Error:", e)


def run_test(users=20):
    threads = []
    start = time.time()

    for i in range(users):
        t = threading.Thread(target=send_request, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("Total Time:", round(time.time() - start, 2))


if __name__ == "__main__":
    run_test(20)
>>>>>>> 0bc0fbb9e046b7272719db153b92df3a456642a2
