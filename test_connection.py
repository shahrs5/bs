import urllib.request
import os
import time

key = os.environ.get("OPENROUTER_API_KEY", "")
print(f"Key length: {len(key)}, starts with: {key[:15]}...")

# Test 1: auth endpoint (GET)
print("\n--- Test 1: Auth check ---")
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/auth/key",
    headers={"Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        print("SUCCESS:", resp.read().decode()[:200])
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: chat completion (POST)
print("\n--- Test 2: Chat completion ---")
import json

payload = json.dumps({
    "model": "anthropic/claude-haiku-4.5",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 10,
}).encode("utf-8")

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=payload,
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
    method="POST",
)
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"SUCCESS ({time.time() - start:.1f}s):", resp.read().decode()[:300])
except Exception as e:
    print(f"FAILED ({time.time() - start:.1f}s): {e}")
