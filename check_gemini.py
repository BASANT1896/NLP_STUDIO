"""Find out exactly why Gemini isn't working.   Run:   python check_gemini.py

Prints the SDK version, which models your key can list, and the real result of a tiny
test call on each candidate model. Your key is never printed in full.
"""
from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv

load_dotenv()

from google import genai  # noqa: E402

import gemini_client as gem  # noqa: E402

key = os.getenv("GEMINI_API_KEY", "").strip()
try:
    print("google-genai version:", version("google-genai"))
except PackageNotFoundError:
    print("google-genai version: unknown")

if not key:
    raise SystemExit("GEMINI_API_KEY is empty. Put it in your .env file (same folder as this script).")
print(f"Key found: starts with '{key[:3]}', {len(key)} characters")
if key.startswith("AQ."):
    print("  (new 'AQ.' auth-key format: keep google-genai up to date with: pip install -U google-genai)")

client = genai.Client(api_key=key)

print("\n1) Models this key can list:")
try:
    listed = gem.list_models(client)
    print("   " + (", ".join(listed) if listed else "none returned"))
except Exception as e:  # noqa: BLE001
    print(f"   Listing failed [{gem._classify(e)}]: {gem._short(e)}")

print("\n2) Test call on each model the app would try (in order):")
working = None
for model in gem.candidate_models(key):
    try:
        resp = client.models.generate_content(model=model, contents="Reply with the single word: ok")
        print(f"   OK      {model}  -> {(resp.text or '').strip()[:20]!r}")
        working = working or model
    except Exception as e:  # noqa: BLE001
        print(f"   FAILED  {model}  [{gem._classify(e)}] {gem._short(e)}")

print()
if working:
    print(f"Result: your key works. To pin it, add this line to .env:  GEMINI_MODEL={working}")
else:
    print("Result: no model worked. Read the [kind] above:")
    print("  auth    -> key rejected (check the key; try a newly created one; update google-genai)")
    print("  quota   -> free limit reached (wait, or retry after midnight Pacific time)")
    print("  missing -> model not available to this key (use a model from list 1)")
