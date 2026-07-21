"""Minimal load test: concurrent requests against a running instance.

Usage:
    python loadtest.py --url http://localhost:8080 --requests 500 --concurrency 20
Reports throughput, latency percentiles, and estimated Cloud Run cost per 1000 requests.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

QUERIES = [
    "günstige Windeln",
    "I need stuff for a pasta dinner",
    "Suche Shampoo bei dm",
    "Bio-Olivenöl oder normales?",
    "Problem mit meinen Punkten",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    latencies = []
    errors = 0

    def one(i: int) -> None:
        nonlocal errors
        query = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        try:
            r = client.post(f"{args.url}/assist", json={"query": query}, timeout=15)
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            errors += 1

    with httpx.Client() as client:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(one, range(args.requests)))
        wall = time.perf_counter() - start

    if not latencies:
        print("All requests failed — is the server running?")
        return

    latencies.sort()
    p = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]
    rps = len(latencies) / wall
    print(f"requests={args.requests} concurrency={args.concurrency} errors={errors}")
    print(f"wall={wall:.2f}s  throughput={rps:.0f} req/s")
    print(f"latency ms: p50={p(0.50):.1f}  p95={p(0.95):.1f}  p99={p(0.99):.1f}  mean={statistics.mean(latencies):.1f}")

    # Cloud Run (europe-west3, tier 1): 1 vCPU + 512 MiB, request-based billing.
    # vCPU $0.000024/s + memory $0.0000025/GiB-s + $0.40 per 1M requests.
    per_request_s = 1 / rps  # billed time amortized at observed throughput
    cpu = 0.000024 * per_request_s
    mem = 0.0000025 * 0.5 * per_request_s
    req = 0.40 / 1_000_000
    print(f"est. Cloud Run cost per 1000 requests (1 vCPU/512MiB at this throughput): "
          f"${(cpu + mem + req) * 1000:.5f}")


if __name__ == "__main__":
    main()
