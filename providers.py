"""Non-Gemini providers used by NLP Studio.

* Groq (free tier)  -> text generation (Llama 3.3 70B by default)
* DeepL API Free    -> translation (best quality, ~500k characters per month)
* NLLB-200 (local)  -> translation fallback: no key, no quota, 200 languages

Groq and DeepL are called with plain `requests`, so no extra SDKs are needed.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

import requests
import streamlit as st


class ProviderError(Exception):
    """Raised with a message that is safe to show to the user."""


def _err_msg(r: requests.Response) -> str:
    try:
        j = r.json()
        e = j.get("error")
        if isinstance(e, dict):
            return str(e.get("message", ""))
        return str(e or j.get("message", "")) or "(empty JSON message)"
    except ValueError:
        return r.text[:200] or f"(empty body from {r.request.method} {r.url})"


# =========================================================== Groq (generation)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Free model rosters change often. Set GROQ_MODEL in .env to force one first.
GROQ_FALLBACK_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]


def _groq_models() -> list[str]:
    return list(dict.fromkeys(m for m in [os.getenv("GROQ_MODEL"), *GROQ_FALLBACK_MODELS] if m))


def groq_generate(api_key: str, prompt: str, system: str | None = None,
                  temperature: float = 0.8, max_tokens: int = 1200) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    last = "No Groq model could be reached."
    for model in _groq_models():
        for attempt in range(2):
            try:
                r = requests.post(GROQ_URL, headers=headers, timeout=90, json={
                    "model": model, "messages": messages,
                    "temperature": temperature, "max_completion_tokens": max_tokens,
                })
            except requests.RequestException as e:
                raise ProviderError(f"Couldn't reach Groq: {e}") from e

            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if text:
                    return text
                last = "Groq returned an empty response."
                break
            if r.status_code in (401, 403):
                raise ProviderError("Groq rejected the API key. Check GROQ_API_KEY.")
            if r.status_code == 429:
                if attempt == 0:
                    try:
                        wait = float(r.headers.get("retry-after", 5))
                    except ValueError:
                        wait = 5.0
                    time.sleep(min(wait, 15))
                    continue
                last = "Groq's free rate limit was reached."
                break
            if r.status_code >= 500 and attempt == 0:
                time.sleep(3)
                continue
            # 400 / 404 / 413 ...: model unavailable or request too large for this model -> try the next model
            last = f"Groq error {r.status_code}: {_err_msg(r)[:200]}"
            break
    raise ProviderError(last)


# =========================================================== DeepL (translation)
# Candidate DeepL target codes per language; the first one DeepL reports as supported is used.
DEEPL_CODES = {
    "English": ["EN-US", "EN"], "Spanish": ["ES"], "French": ["FR"], "German": ["DE"], "Italian": ["IT"],
    "Portuguese": ["PT-BR", "PT"], "Dutch": ["NL"], "Russian": ["RU"], "Arabic": ["AR"],
    "Chinese (Simplified)": ["ZH-HANS", "ZH"], "Japanese": ["JA"], "Korean": ["KO"], "Hindi": ["HI"],
    "Bengali": ["BN"], "Tamil": ["TA"], "Telugu": ["TE"], "Marathi": ["MR"], "Urdu": ["UR"],
    "Turkish": ["TR"], "Indonesian": ["ID"], "Vietnamese": ["VI"], "Thai": ["TH"], "Polish": ["PL"],
    "Swedish": ["SV"], "Greek": ["EL"],
}
_deepl_lang_cache: dict[str, frozenset] = {}


def _deepl_base(key: str) -> str:
    return "https://api-free.deepl.com" if key.strip().endswith(":fx") else "https://api.deepl.com"


def _deepl_headers(key: str) -> dict:
    return {"Authorization": f"DeepL-Auth-Key {key.strip()}"}


def _deepl_targets(key: str) -> frozenset:
    """Ask DeepL which target languages this key can use (cached after the first success)."""
    if key in _deepl_lang_cache:
        return _deepl_lang_cache[key]
    try:
        r = requests.get(f"{_deepl_base(key)}/v2/languages", params={"type": "target"},
                         headers=_deepl_headers(key), timeout=20)
    except requests.RequestException as e:
        raise ProviderError(f"Couldn't reach DeepL: {e}") from e
    if r.status_code in (401, 403):
        raise ProviderError("DeepL rejected the API key. Check DEEPL_API_KEY.")
    if r.status_code != 200:
        raise ProviderError(f"DeepL error {r.status_code}: {_err_msg(r)[:200]}")
    langs = frozenset(x["language"].upper() for x in r.json())
    _deepl_lang_cache[key] = langs
    return langs


def deepl_code(key: str, language: str) -> str | None:
    """DeepL code for `language`, or None if DeepL doesn't offer it."""
    candidates = DEEPL_CODES.get(language, [])
    try:
        supported = _deepl_targets(key)
    except ProviderError as e:
        if "rejected the API key" in str(e):
            raise
        return candidates[0] if candidates else None  # lookup failed; trust the static table
    for c in candidates:
        if c in supported:
            return c
    return None


def deepl_translate(key: str, chunks: list[str], code: str,
                    on_progress: Callable[[float], None] | None = None) -> str:
    # --- SANITY CHECK START ---
    masked_key = f"{key[:5]}...{key[-5:]}" if len(key) > 10 else "KEY_TOO_SHORT"
    base_url = _deepl_base(key)
    print("\n" + "=" * 50)
    print(f"[SANITY CHECK] Initiating DeepL Translation Call")
    print(f"[SANITY CHECK] Base URL: {base_url}")
    print(f"[SANITY CHECK] Key Suffix FX: {key.strip().endswith(':fx')}")
    print(f"[SANITY CHECK] Key Preview: {masked_key}")
    print(f"[SANITY CHECK] Chunks to process: {len(chunks)}")
    print("=" * 50)
    # --- SANITY CHECK END ---

    url = f"{base_url}/v2/translate"
    out: list[str] = []

    # 1. Normalize target language code
    target_code = code.upper()
    if target_code == "EN":
        target_code = "EN-US"
    elif target_code == "PT":
        target_code = "PT-BR"

    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue

        for attempt in range(3):
            try:
                # 2. Schema-compliant payload
                payload = {
                    "text": [chunk],
                    "target_lang": target_code,
                    "split_sentences": "0",       # Use "0" or "1" (string) for plain text
                    "preserve_formatting": True   # Strict boolean True
                }

                print(f"[SANITY CHECK] Chunk {i}/{len(chunks)} -> Sending {len(chunk)} chars to DeepL...")
                r = requests.post(url, headers=_deepl_headers(key), timeout=60, json=payload)
                print(f"[SANITY CHECK] DeepL HTTP Status: {r.status_code}")

            except requests.RequestException as e:
                print(f"[SANITY CHECK ERROR] Network exception: {e}")
                raise ProviderError(f"Couldn't reach DeepL: {e}") from e

            if r.status_code == 200:
                res_text = r.json()["translations"][0]["text"]
                out.append(res_text)
                print(f"[SANITY CHECK SUCCESS] Received translation: '{res_text[:40]}...'")

                # Fetch real-time character usage directly from DeepL API
                try:
                    usage_r = requests.get(f"{base_url}/v2/usage", headers=_deepl_headers(key), timeout=10)
                    if usage_r.status_code == 200:
                        u = usage_r.json()
                        print(f"[SANITY CHECK USAGE] Live API Character Count: {u.get('character_count')}/{u.get('character_limit')}")
                except Exception as usage_err:
                    print(f"[SANITY CHECK WARNING] Could not fetch live usage: {usage_err}")

                break
            if r.status_code in (401, 403):
                raise ProviderError("DeepL rejected the API key. Check DEEPL_API_KEY.")
            if r.status_code == 456:
                raise ProviderError("The monthly DeepL free character quota is used up.")
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue

            # Display the exact API message on error to avoid silent failures
            err_details = r.text
            try:
                err_details = r.json().get("message", r.text)
            except Exception:
                pass

            raise ProviderError(
                f"DeepL error {r.status_code} on {r.request.method} {r.url}: "
                f"{r.text[:300] or '(empty body)'}"
            )

        if on_progress:
            on_progress(i / len(chunks))

    return "\n\n".join(out)

# =========================================================== NLLB-200 (local translation)
NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")  # CC-BY-NC: non-commercial use only
NLLB_CODES = {
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn", "German": "deu_Latn",
    "Italian": "ita_Latn", "Portuguese": "por_Latn", "Dutch": "nld_Latn", "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab", "Chinese (Simplified)": "zho_Hans", "Japanese": "jpn_Jpan", "Korean": "kor_Hang",
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Marathi": "mar_Deva", "Urdu": "urd_Arab", "Turkish": "tur_Latn", "Indonesian": "ind_Latn",
    "Vietnamese": "vie_Latn", "Thai": "tha_Thai", "Polish": "pol_Latn", "Swedish": "swe_Latn",
    "Greek": "ell_Grek",
}
# langdetect code -> NLLB code, used to tell NLLB what language the source text is in
_DETECT_TO_NLLB = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab", "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant", "ja": "jpn_Jpan", "ko": "kor_Hang", "hi": "hin_Deva", "bn": "ben_Beng",
    "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva", "ur": "urd_Arab", "tr": "tur_Latn",
    "id": "ind_Latn", "vi": "vie_Latn", "th": "tha_Thai", "pl": "pol_Latn", "sv": "swe_Latn",
    "el": "ell_Grek", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
    "ne": "npi_Deva", "uk": "ukr_Cyrl", "he": "heb_Hebr", "fa": "pes_Arab", "cs": "ces_Latn",
    "ro": "ron_Latn", "hu": "hun_Latn", "da": "dan_Latn", "fi": "fin_Latn", "no": "nob_Latn",
    "bg": "bul_Cyrl",
}
_NO_SPACE_TARGETS = {"zho_Hans", "jpn_Jpan", "tha_Thai"}


def _detect_source(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return _DETECT_TO_NLLB.get(detect(text[:3000]), "eng_Latn")
    except Exception:  # noqa: BLE001
        return "eng_Latn"


@st.cache_resource(show_spinner=False)
def _nllb_model():
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.eval()
    return model


def _sentence_chunks(par: str, max_chars: int = 450) -> list[str]:
    par = re.sub(r"\s+", " ", par).strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|(?<=[。！？])", par) if p and p.strip()]
    out: list[str] = []
    cur = ""
    for s in parts:
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:cut])
            s = s[cut:].lstrip()
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def nllb_translate(text: str, language: str, on_progress: Callable[[float], None] | None = None) -> str:
    tgt = NLLB_CODES.get(language)
    if not tgt:
        raise ProviderError(f"The local model doesn't support {language}.")
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang=_detect_source(text))
    model = _nllb_model()
    forced = tok.convert_tokens_to_ids(tgt)

    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    jobs = [(pi, s) for pi, p in enumerate(paragraphs) for s in _sentence_chunks(p)]
    if not jobs:
        return ""
    translated: list[str] = [""] * len(jobs)
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][1]))  # similar lengths batch faster
    batch = 8
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        enc = tok([jobs[i][1] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=forced, max_new_tokens=256, num_beams=2)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            translated[i] = t
        if on_progress:
            on_progress(min(1.0, (start + batch) / len(order)))

    joiner = "" if tgt in _NO_SPACE_TARGETS else " "
    per_par: list[list[str]] = [[] for _ in paragraphs]
    for (pi, _), t in zip(jobs, translated):
        per_par[pi].append(t)
    return "\n\n".join(joiner.join(p) for p in per_par)