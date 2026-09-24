import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("DEEPL_API_KEY", "").strip()
print("key length:", len(key), "| ends with :fx ->", key.endswith(":fx"))
base = "https://api-free.deepl.com" if key.endswith(":fx") else "https://api.deepl.com"
h = {"Authorization": f"DeepL-Auth-Key {key}"}

r = requests.get(f"{base}/v2/languages", params={"type": "target"}, headers=h)
print("languages:", r.status_code, r.text[:100])

r = requests.post(f"{base}/v2/translate", headers=h,
                  json={"text": ["Hello world"], "target_lang": "ES"})
print("translate:", r.status_code, r.text)