"""Thin wrapper around the google-genai SDK (Gemini free tier from Google AI Studio).

Google renames and retires free-tier models often, so the model list is not hard-coded:

1. GEMINI_MODEL from .env (if set) is always tried first.
2. Otherwise the app asks your key which models it can use (client.models.list()),
   keeps the text-capable Flash models, and orders them Flash-Lite first (biggest free
   quota), stable before preview, newest first.
3. If listing fails, a built-in fallback list is used.

Each model has its own free quota, so a rate-limited model moves on to the next one.
Errors from every model that was tried are kept, so the message you see explains what
really went wrong (bad key, quota, missing model) instead of only the last failure.
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Callable, Iterator

from google import genai
from google.genai import types

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
]
_SKIP = ("image", "tts", "live", "audio", "transcribe", "translate", "embed", "robotics",
         "omni", "computer", "cyber", "learnlm", "gemma", "vision", "thinking", "-exp")
MAX_MODELS_TRIED = 6

_working: dict[str, str] = {}      # api_key -> model that last worked
_discovered: dict[str, list[str]] = {}  # api_key -> models the key can list


class GeminiError(Exception):
    """Every candidate model failed. `kind` is auth | quota | busy | missing | other."""

    def __init__(self, kind: str, attempts: dict[str, str]):
        self.kind = kind
        self.attempts = attempts
        super().__init__(_summary(kind, attempts))


@lru_cache(maxsize=4)
def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# ------------------------------------------------------------------ model discovery
def list_models(client: genai.Client) -> list[str]:
    """Text-capable Flash models this key can use. Raises if the key/listing fails."""
    names = []
    for m in client.models.list():
        name = (getattr(m, "name", "") or "").replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if not name.startswith("gemini-") or "flash" not in name:
            continue
        if any(s in name for s in _SKIP):
            continue
        if actions and "generateContent" not in actions:
            continue
        names.append(name)
    return sorted(set(names), key=_rank)


def _rank(name: str):
    m = re.match(r"gemini-([\d.]+)", name)
    ver = tuple(-int(x) for x in m.group(1).split(".") if x.isdigit()) if m else (0,)
    return ("lite" not in name, "preview" in name, ver, name)


def candidate_models(api_key: str) -> list[str]:
    """Ordered list of models to try for this key."""
    if api_key not in _discovered:
        try:
            found = list_models(_client(api_key))
            if found:
                _discovered[api_key] = found
        except Exception:  # noqa: BLE001  (listing is best-effort; fall back to the built-in list)
            pass
    base = _discovered.get(api_key) or FALLBACK_MODELS
    models = list(dict.fromkeys(m for m in [os.getenv("GEMINI_MODEL"), *base] if m))
    good = _working.get(api_key)
    if good in models:
        models.remove(good)
        models.insert(0, good)
    return models[:MAX_MODELS_TRIED]


# ------------------------------------------------------------------ error handling
def _short(e: Exception) -> str:
    return re.sub(r"\s+", " ", str(e))[:200]


def _classify(e: Exception) -> str:
    code = getattr(e, "code", None)
    ml = str(e).lower()
    if (code in (401, 403) or "api key not valid" in ml or "api_key_invalid" in ml
            or "access_token_type_unsupported" in ml or "permission_denied" in ml or "unauthenticated" in ml):
        return "auth"
    if code == 429 or "resource_exhausted" in ml:
        return "quota"
    if code == 404 or "not_found" in ml or "is not found for api version" in ml:
        return "missing"
    if code in (500, 502, 503, 504) or "unavailable" in ml or "overloaded" in ml:
        return "busy"
    if code == 400 and "temperature" in ml:
        return "temperature"
    return "other"


def _is_daily_quota(e: Exception) -> bool:
    ml = str(e).lower()
    return "perday" in ml or "per day" in ml or "daily" in ml


def _overall(attempts: dict[str, str]) -> str:
    kinds = [v.split(":")[0] for v in attempts.values()]
    if "quota" in kinds:
        return "quota"
    if "busy" in kinds:
        return "busy"
    if kinds and all(k == "missing" for k in kinds):
        return "missing"
    return "other"


def _summary(kind: str, attempts: dict[str, str]) -> str:
    tried = ", ".join(f"{m} ({v.split(':')[0]})" for m, v in attempts.items())
    if kind == "auth":
        return ("Gemini rejected the API key. Check GEMINI_API_KEY in .env. If the key starts with 'AQ.', "
                "update the SDK (pip install -U google-genai) and try a freshly created key. "
                "Run 'python check_gemini.py' for details.")
    if kind == "quota":
        return ("The free Gemini quota is used up for now (per-minute or per-day limit). Wait a little, or try "
                "again after midnight Pacific time if it's the daily limit. "
                f"Models tried: {tried}. Run 'python check_gemini.py' for details.")
    if kind == "busy":
        return f"Gemini is overloaded right now. Try again in a minute. Models tried: {tried}."
    if kind == "missing":
        return ("None of the Gemini models the app tried is available to your key. "
                f"Models tried: {tried}. Run 'python check_gemini.py' to see which models your key can use, "
                "then set GEMINI_MODEL in .env to one of them.")
    detail = "; ".join(f"{m}: {v}" for m, v in attempts.items())
    return f"Gemini error: {detail[:400]}"


def friendly_error(e: Exception | None) -> str:
    if isinstance(e, GeminiError):
        return str(e)
    if e is None:
        return "Unknown error"
    kind = _classify(e)
    if kind == "other":
        return f"Something went wrong: {_short(e)}"
    return _summary(kind, {"gemini": f"{kind}: {_short(e)}"})


# ------------------------------------------------------------------ calls
def _cfg(system: str | None, temperature: float | None) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(system_instruction=system, temperature=temperature)


def _run(api_key: str, call: Callable[[str, bool], object]):
    """Try each candidate model until `call(model, use_temperature)` succeeds."""
    attempts: dict[str, str] = {}
    for model in candidate_models(api_key):
        use_temp = True
        for attempt in range(3):
            try:
                result = call(model, use_temp)
                _working[api_key] = model
                return result
            except Exception as e:  # noqa: BLE001
                kind = _classify(e)
                if kind == "auth":
                    raise GeminiError("auth", {model: f"auth: {_short(e)}"}) from e
                if kind == "temperature" and use_temp:
                    use_temp = False  # newer models deprecate sampling parameters
                    continue
                if kind == "busy" and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                if kind == "quota" and attempt == 0 and not _is_daily_quota(e):
                    time.sleep(5)  # per-minute limit: one short wait, then move on
                    continue
                attempts[model] = f"{kind}: {_short(e)}"
                break
    raise GeminiError(_overall(attempts), attempts)


def generate(api_key: str, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
    """One-shot text generation. Raises GeminiError on failure."""
    client = _client(api_key)

    def call(model: str, use_temp: bool) -> str:
        resp = client.models.generate_content(
            model=model, contents=prompt, config=_cfg(system, temperature if use_temp else None))
        text = resp.text or ""
        if not text.strip():
            raise RuntimeError("Gemini returned an empty response (it may have been blocked by a safety filter).")
        return text

    return _run(api_key, call)


def stream(api_key: str, history: list[dict], system: str, temperature: float = 0.7) -> Iterator[str]:
    """Stream a chat reply. `history` is [{'role': 'user'|'assistant', 'content': str}, ...]."""
    client = _client(api_key)
    contents = [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])])
        for m in history
    ]

    def call(model: str, use_temp: bool):
        it = iter(client.models.generate_content_stream(
            model=model, contents=contents, config=_cfg(system, temperature if use_temp else None)))
        for chunk in it:  # errors surface on the first read, so read it inside the retry loop
            if chunk.text:
                return chunk.text, it
        raise RuntimeError("Gemini returned an empty response (it may have been blocked by a safety filter).")

    try:
        first, rest = _run(api_key, call)
    except GeminiError as e:
        yield f"⚠️ {e}"
        return
    yield first
    try:
        for chunk in rest:
            if chunk.text:
                yield chunk.text
    except Exception as e:  # noqa: BLE001
        yield f"\n\n⚠️ {friendly_error(e)}"
