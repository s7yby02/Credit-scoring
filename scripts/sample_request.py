"""Send real accounts to the scoring API and print the answer + latency.

Usage: python scripts/sample_request.py data/prod 50
"""
import json
import statistics
import sys
import time
from pathlib import Path

import requests

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/prod")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
apps = {a["account_id"]: a for a in map(json.loads, open(folder / "applications.jsonl"))}

latencies = []
with open(folder / "accounts.jsonl") as f:
    for i, line in enumerate(f):
        if i == n:
            break
        account = json.loads(line)
        application = apps[account["_id"]]
        payload = {
            "application_id": application["_id"],
            "application_date": application["application_date"],
            "transactions": account["transactions"],
        }
        start = time.perf_counter()
        response = requests.post("http://localhost:8000/score", json=payload, timeout=10)
        latencies.append((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        if i == 0:
            print(json.dumps(response.json(), indent=2))

latencies.sort()
p95 = latencies[int(0.95 * (len(latencies) - 1))]
print(f"{len(latencies)} requests: median {statistics.median(latencies):.0f} ms, p95 {p95:.0f} ms")
