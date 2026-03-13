import urllib.request
import os
import json
import time

key = os.environ.get("OPENROUTER_API_KEY", "")

# Test with the exact same payload the benchmark sends
payload = json.dumps({
    "model": "anthropic/claude-haiku-4.5",
    "messages": [{"role": "user", "content": "Say hi"}],
    "reasoning": {"effort": "none"},
    "provider": {"require_parameters": True},
}).encode("utf-8")

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "X-Title": "bullshit-benchmark",
}

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=payload,
    headers=headers,
    method="POST",
)

print("Sending request with reasoning + provider params...")
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"SUCCESS ({time.time() - start:.1f}s):", resp.read().decode()[:300])
except Exception as e:
    print(f"FAILED ({time.time() - start:.1f}s): {e}")
