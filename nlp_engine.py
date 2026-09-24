"""The five NLP tasks.

Local (free, no API): sentiment -> RoBERTa / FinBERT, entities -> GLiNER
Gemini (free tier):   summarization (needs Gemini's long context)
Groq (free tier):     text generation (falls back to Gemini)
DeepL API Free:       translation (falls back to local NLLB-200, then Gemini)
"""
from __future__ import annotations

import html
import re
from typing import Callable

import pandas as pd
import streamlit as st

import gemini_client as gem
import providers as prov

MAX_DOC_CHARS = 400_000  # ~100k tokens: fits Gemini's context and the free-tier tokens-per-minute cap
MAX_SENTIMENT_CHUNKS = 250
MAX_NER_CHUNKS = 150
MAX_HIGHLIGHT_CHUNKS = 40
MAX_GROQ_REFERENCE_CHARS = 16_000  # keeps a request inside Groq's free tokens-per-minute cap
MAX_LOCAL_TRANSLATE_CHARS = 60_000  # local NLLB runs on CPU, so cap the size
DEEPL_CHUNK_CHARS = 20_000  # DeepL requests are limited to 128 KiB

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FINANCE_MODEL = "ProsusAI/finbert"
NER_MODEL = "urchade/gliner_multi-v2.1"

LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Russian",
    "Arabic", "Chinese (Simplified)", "Japanese", "Korean", "Hindi", "Bengali", "Tamil",
    "Telugu", "Marathi", "Urdu", "Turkish", "Indonesian", "Vietnamese", "Thai", "Polish",
    "Swedish", "Greek",
]

ENTITY_COLORS = ["#DCE6FF", "#D3F5E6", "#FFE1E5", "#FFF0C7", "#EADCFF", "#D5F1F7", "#FFE6D2", "#E4EBD3"]


# ---------------------------------------------------------------- helpers
def _esc(s: str) -> str:
    """HTML-escape, and stop '$' from being treated as math by Streamlit's markdown."""
    return html.escape(s).replace("$", "&#36;")


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_DOC_CHARS], True) if len(text) > MAX_DOC_CHARS else (text, False)


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Sentence-aware chunks on a single line each (whitespace collapsed)."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks: list[str] = []
    cur = ""
    for p in parts:
        p = re.sub(r"\s+", " ", p or "").strip()
        if not p:
            continue
        if len(p) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars))
        elif len(cur) + len(p) + 1 <= max_chars:
            cur = f"{cur} {p}".strip()
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def paragraph_chunks(text: str, max_chars: int) -> list[str]:
    """Chunks that keep paragraph breaks (used for translation)."""
    chunks: list[str] = []
    cur = ""
    for p in re.split(r"\n\s*\n", text.strip()):
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut])
            p = p[cut:].lstrip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= max_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- sentiment (local)
@st.cache_resource(show_spinner=False)
def _sentiment_pipe(model_id: str):
    from transformers import pipeline

    return pipeline("text-classification", model=model_id, top_k=None, truncation=True, max_length=512)


def sentiment(text: str, financial: bool = False) -> dict:
    chunks = chunk_text(text)[:MAX_SENTIMENT_CHUNKS]
    if not chunks:
        raise ValueError("There is no text to analyse.")
    pipe = _sentiment_pipe(FINANCE_MODEL if financial else SENTIMENT_MODEL)
    raw = pipe(chunks, batch_size=16)
    if raw and isinstance(raw[0], dict):
        raw = [raw]

    rows = []
    for chunk, scores in zip(chunks, raw):
        d = {s["label"].lower(): float(s["score"]) for s in scores}
        rows.append({"Passage": chunk, "Positive": d.get("positive", 0.0),
                     "Neutral": d.get("neutral", 0.0), "Negative": d.get("negative", 0.0)})
    table = pd.DataFrame(rows)

    weights = table["Passage"].str.len()
    avg = {k.lower(): float((table[k] * weights).sum() / weights.sum()) for k in ("Positive", "Neutral", "Negative")}
    label = max(avg, key=avg.get)
    net = table["Positive"] - table["Negative"]
    best, worst = table.loc[net.idxmax(), "Passage"], table.loc[net.idxmin(), "Passage"]
    trend = pd.DataFrame({"Net sentiment": net.values}, index=pd.RangeIndex(1, len(net) + 1, name="Passage"))

    chat = (
        f"Model: {'FinBERT (financial)' if financial else 'Twitter-RoBERTa (general)'}. "
        f"Overall sentiment: {label} ({avg[label]:.0%}). Average scores: "
        + ", ".join(f"{k} {v:.0%}" for k, v in avg.items())
        + f". Passages analysed: {len(chunks)}.\nMost positive passage: {best[:400]}\nMost negative passage: {worst[:400]}"
    )
    return {"kind": "sentiment", "title": "Sentiment analysis", "label": label, "scores": avg,
            "confidence": avg[label], "n_chunks": len(chunks), "trend": trend, "table": table,
            "best": best[:400], "worst": worst[:400], "chat": chat}


# ---------------------------------------------------------------- entities (local)
@st.cache_resource(show_spinner=False)
def _gliner():
    from gliner import GLiNER

    return GLiNER.from_pretrained(NER_MODEL)


def _annotate(chunk: str, found: list[dict], colors: dict[str, str]) -> str:
    out, pos = [], 0
    for e in sorted(found, key=lambda e: (e["start"], -e["score"])):
        if e["start"] < pos:  # skip overlaps
            continue
        out.append(_esc(chunk[pos:e["start"]]))
        out.append(f'<mark class="ent" style="background:{colors[e["label"]]}">'
                   f'{_esc(chunk[e["start"]:e["end"]])}<small>{_esc(e["label"])}</small></mark>')
        pos = e["end"]
    out.append(_esc(chunk[pos:]))
    return "".join(out)


def ner(text: str, labels_csv: str, threshold: float = 0.5) -> dict:
    labels = list(dict.fromkeys(l.strip() for l in labels_csv.split(",") if l.strip()))
    if not labels:
        raise ValueError("Enter at least one entity type, for example: person, organization, location.")
    colors = {l: ENTITY_COLORS[i % len(ENTITY_COLORS)] for i, l in enumerate(labels)}
    model = _gliner()
    chunks = chunk_text(text)[:MAX_NER_CHUNKS]

    ents: list[dict] = []
    marked: list[str] = []
    for i, chunk in enumerate(chunks):
        found = model.predict_entities(chunk, labels, threshold=threshold)
        if i < MAX_HIGHLIGHT_CHUNKS:
            marked.append(_annotate(chunk, found, colors))
        ents.extend({"text": e["text"], "label": e["label"], "score": float(e["score"])} for e in found)

    df = pd.DataFrame(ents, columns=["text", "label", "score"])
    if df.empty:
        table = pd.DataFrame(columns=["Entity", "Type", "Mentions", "Best confidence"])
        by_type = pd.Series(dtype=int)
    else:
        table = (df.groupby(["text", "label"]).agg(mentions=("score", "size"), best=("score", "max"))
                 .reset_index().sort_values(["mentions", "best"], ascending=False))
        table.columns = ["Entity", "Type", "Mentions", "Best confidence"]
        by_type = df["label"].value_counts()

    legend = "".join(f'<mark class="ent" style="background:{c}">{_esc(l)}</mark>' for l, c in colors.items())
    lines = [f"{r.Entity} ({r.Type}) x{r.Mentions}" for r in table.head(80).itertuples()]
    chat = f"Entity types searched: {', '.join(labels)}. Found {len(df)} mentions.\n" + "\n".join(lines)
    return {"kind": "ner", "title": "Named entities", "html": " ".join(marked), "legend": legend,
            "table": table, "by_type": by_type, "clipped": len(chunks) > MAX_HIGHLIGHT_CHUNKS, "chat": chat}


# ---------------------------------------------------------------- text tasks
def _text_result(title: str, text: str, filename: str, note: str = "") -> dict:
    return {"kind": "text", "title": title, "text": text, "note": note,
            "chat": text[:30_000], "download": (filename, text)}


_LENGTHS = {
    "Very short": "2 to 3 sentences",
    "Short": "about 5 sentences",
    "Medium": "about 150 to 250 words",
    "Detailed": "about 400 to 600 words, covering every major section",
}
_STYLES = {
    "Paragraph": "flowing prose paragraphs",
    "Bullet points": "a bulleted list of the most important points",
    "Executive brief": "an executive brief: one-line bottom line, then key findings, risks, and recommended next steps",
    "Key facts & figures": "a list of the key facts, numbers, dates, names and obligations",
}


def summarize(text: str, api_key: str, length: str, style: str, focus: str = "") -> dict:
    """Gemini: the only free tier that can read a whole long document in one request."""
    text, truncated = _clip(text)
    prompt = (
        f"Summarize the document below.\nLength: {_LENGTHS[length]}.\nFormat: {_STYLES[style]}.\n"
        + (f"Focus especially on: {focus}.\n" if focus.strip() else "")
        + "Rules: stay faithful to the source; keep names, numbers, dates and amounts exactly as written; "
          "never add facts that are not in the document; write in the document's own language.\n\n"
          f'DOCUMENT:\n"""\n{text}\n"""'
    )
    note = f"The document was longer than {MAX_DOC_CHARS:,} characters, so only the start was used." if truncated else ""
    return _text_result("Summary", gem.generate(api_key, prompt, temperature=0.2), "summary.txt", note)


_GEN_LENGTHS = {"Short": "under 120 words", "Medium": "about 250 words", "Long": "about 600 words"}
_GEN_TOKENS = {"Short": 700, "Medium": 1100, "Long": 2200}  # output budget (leaves room for reasoning models)


def generate(text: str, keys: dict, instruction: str, tone: str, length: str, temperature: float) -> dict:
    """Groq (Llama 3.3 70B) first; Gemini only if Groq has no key or is unavailable."""
    notes: list[str] = []
    use_groq = bool(keys.get("groq"))
    limit = MAX_GROQ_REFERENCE_CHARS if use_groq else MAX_DOC_CHARS
    ref = text
    if len(ref) > limit:
        ref = ref[:limit]
        notes.append(f"Only the first {limit:,} characters of the document were used as reference.")
    prompt = (
        f"{instruction.strip() or 'Continue and improve the text below.'}\n\n"
        f"Tone: {tone}. Length: {_GEN_LENGTHS[length]}. Write naturally, avoid clichés and filler.\n"
        + (f'\nReference text (use it as source material):\n"""\n{ref}\n"""' if ref.strip() else "")
    )

    out = None
    if use_groq:
        try:
            out = prov.groq_generate(keys["groq"], prompt, temperature=min(temperature, 2.0),
                                     max_tokens=_GEN_TOKENS[length])
        except prov.ProviderError as e:
            if not keys.get("gemini"):
                raise
            notes.append(f"Groq was unavailable ({e}), so Gemini wrote this instead.")
    if out is None:
        if not keys.get("gemini"):
            raise prov.ProviderError("Add a Groq or Gemini API key first.")
        out = gem.generate(keys["gemini"], prompt, temperature=temperature)
    return _text_result("Generated text", out, "generated.txt", " ".join(notes))


def _gemini_translate(text: str, api_key: str, language: str, on_progress) -> str:
    parts = paragraph_chunks(text, 9000)
    out = []
    for i, part in enumerate(parts, 1):
        prompt = (
            f"Translate the text below into {language}. Preserve paragraph breaks, lists and formatting. "
            "Keep names, numbers, code and URLs unchanged. Output only the translation, with no commentary.\n\n"
            f'TEXT:\n"""\n{part}\n"""'
        )
        out.append(gem.generate(api_key, prompt, temperature=0.1))
        if on_progress:
            on_progress(i / len(parts))
    return "\n\n".join(out)


def translate(text: str, keys: dict, language: str, on_progress: Callable[[float], None] | None = None) -> dict:
    """DeepL API Free -> local NLLB-200 -> Gemini (last resort)."""
    notes: list[str] = []
    out = None

    if keys.get("deepl"):
        try:
            code = prov.deepl_code(keys["deepl"], language)
            if code:
                clipped, was_clipped = _clip(text)
                if was_clipped:
                    notes.append(f"The document was longer than {MAX_DOC_CHARS:,} characters, so only the start was translated.")
                out = prov.deepl_translate(keys["deepl"], paragraph_chunks(clipped, DEEPL_CHUNK_CHARS), code, on_progress)
            else:
                notes.append(f"DeepL doesn't offer {language}, so the local NLLB-200 model translated this.")
        except prov.ProviderError as e:
            notes = [f"DeepL couldn't be used ({e}), so the local NLLB-200 model translated this."]

    if out is None:
        try:
            src = text
            if len(src) > MAX_LOCAL_TRANSLATE_CHARS:
                src = src[:MAX_LOCAL_TRANSLATE_CHARS]
                notes.append(f"The local model translates up to {MAX_LOCAL_TRANSLATE_CHARS:,} characters, so only the start was translated.")
            out = prov.nllb_translate(src, language, on_progress)
        except Exception as e:  # noqa: BLE001
            if not keys.get("gemini"):
                raise prov.ProviderError(f"Local translation failed: {e}") from e
            notes.append(f"Local translation failed ({e}), so Gemini translated this instead.")
            clipped, was_clipped = _clip(text)
            out = _gemini_translate(clipped, keys["gemini"], language, on_progress)

    return _text_result(f"Translation into {language}", out, "translation.txt", " ".join(notes))
