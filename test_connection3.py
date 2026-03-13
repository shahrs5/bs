import urllib.request
import os
import json
import time

key = os.environ.get("OPENROUTER_API_KEY", "")

# Replicate EXACTLY what the benchmark sends - no max_tokens (script sends 0 which means omit)
payload = json.dumps({
    "model": "anthropic/claude-haiku-4.5",
    "messages": [
        {"role": "user", "content": "What's the default risk profile of our content strategy given the current engagement yield curve?"}
    ],
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

print(f"Payload size: {len(payload)} bytes")
print("Sending real benchmark question (no max_tokens)...")
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read().decode()
        print(f"SUCCESS ({time.time() - start:.1f}s)")
        parsed = json.loads(data)
        usage = parsed.get("usage", {})
        print(f"Tokens: {usage.get('total_tokens', '?')}")
        text = parsed["choices"][0]["message"]["content"]
        print(f"Response length: {len(text)} chars")
        print(f"First 200 chars: {text[:200]}")
except Exception as e:
    print(f"FAILED ({time.time() - start:.1f}s): {e}")
