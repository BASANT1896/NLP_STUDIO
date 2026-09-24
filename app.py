"""NLP Studio: a pure-Python Streamlit app.   Run with:   streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


import file_parser  # noqa: E402
import gemini_client as gem  # noqa: E402
import nlp_engine as nlp  # noqa: E402
import providers as prov  # noqa: E402

BASE = Path(__file__).parent
st.set_page_config(page_title="NLP Studio", page_icon="🧠", layout="wide")
st.markdown(f"<style>{(BASE / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("result", None)
ss.setdefault("api_key", "")
ss.setdefault("groq_key", "")
ss.setdefault("deepl_key", "")
ss.setdefault("doc_text", "")
ss.setdefault("doc_name", "Pasted text")

TASKS = {
    "😊 Sentiment analysis": "sentiment",
    "📝 Summarization": "summary",
    "✍️ Text generation": "generate",
    "🏷️ Named entities": "ner",
    "🌍 Translation": "translate",
}
ENGINE_NOTE = {
    "sentiment": "",
    "summary": "",
    "generate": "",
    "ner": "",
    "translate": "",
}
COLORS = {"positive": "#12B886", "neutral": "#8A94A6", "negative": "#F0596B"}

SYSTEM_BASE = (
    "You are the assistant built into NLP Studio, a tool where users analyse documents "
    "(sentiment, summaries, generated text, named entities, translation). "
    "Answer questions about the user's document and about the tool's latest output. "
    "Explain complex or technical passages in plain language, quote the relevant part of the document when helpful, "
    "and say so if the answer is not in the document. For unrelated questions, answer as a helpful general assistant. "
    "Be concise and use short paragraphs or bullets."
)
QUICK_PROMPTS = ["Explain this document in simple terms", "What are the key takeaways?", "Explain the results above"]


def _safe(md: str) -> str:
    """Stop '$' from being rendered as math."""
    return md.replace("$", r"\$")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## NLP Studio")
    st.caption("Pick a task")
    task = TASKS[st.radio("Task", list(TASKS), label_visibility="collapsed")]
    st.divider()

    opts: dict = {}
    if task == "sentiment":
        opts["financial"] = st.radio("Type of text", ["Social View", "Financial View"]) == "Financial View"
    elif task == "summary":
        opts["length"] = st.select_slider("Length", list(nlp._LENGTHS), value="Medium")
        opts["style"] = st.selectbox("Format", list(nlp._STYLES))
        opts["focus"] = st.text_input("Focus on (optional)", placeholder="e.g. payment terms")
    elif task == "generate":
        opts["instruction"] = st.text_area("What would you like?", height=110,
                                           placeholder="e.g. Turn this into a polite email to the client")
        opts["tone"] = st.selectbox("Tone", ["Professional", "Friendly", "Persuasive", "Academic", "Casual", "Concise"])
        opts["length"] = st.select_slider("Length", list(nlp._GEN_LENGTHS), value="Medium")
        opts["temperature"] = st.slider("Creativity", 0.0, 1.5, 0.8, 0.1)
    elif task == "ner":
        opts["labels"] = st.text_input("Entity types, separated by commas",
                                       "person, organization, location, date, money, product")
        opts["threshold"] = st.slider("Confidence threshold", 0.3, 0.9, 0.5, 0.05)
    elif task == "translate":
        opts["language"] = st.selectbox("Translate into", nlp.LANGUAGES, index=1)

    st.divider()
    env_keys = {name: os.getenv(var, "") for name, var in
                (("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY"), ("deepl", "DEEPL_API_KEY"))}
    if env_keys["gemini"]:
        st.success("Perform any Basic NLP Task")
    else:
        st.text_input("Gemini API key", type="password", key="api_key",
                      help="Free key from aistudio.google.com/apikey. Or put it in a .env file.")
    if not (env_keys["groq"] and env_keys["deepl"]):
        with st.expander("Optional keys"):
            if not env_keys["groq"]:
                st.text_input("Groq API key (text generation)", type="password", key="groq_key",
                              help="Free key from console.groq.com/keys. Or put GROQ_API_KEY in .env.")
            if not env_keys["deepl"]:
                st.text_input("DeepL API key (translation)", type="password", key="deepl_key",
                              help="Free key from deepl.com/pro-api. Or put DEEPL_API_KEY in .env.")
    keys = {
        "gemini": env_keys["gemini"] or ss.api_key.strip(),
        "groq": env_keys["groq"] or ss.groq_key.strip(),
        "deepl": env_keys["deepl"] or ss.deepl_key.strip(),
    }
    api_key = keys["gemini"]  # used by the chatbot and summarization
    #st.caption("Local models download once on first use, so the first run of sentiment or entities is slower.")


# ------------------------------------------------------------------ result rendering
def meter(label: str, value: float, color: str) -> str:
    return (f'<div class="meter"><div class="row"><span>{label}</span><span>{value:.0%}</span></div>'
            f'<div class="track"><div class="fill" style="width:{value * 100:.1f}%;background:{color}"></div></div></div>')


def render_sentiment(r: dict) -> None:
    st.markdown(
        f'<div class="verdict"><span class="pill {r["label"]}">{r["label"].title()}</span>'
        f'<span class="muted">{r["confidence"]:.0%} confidence across {r["n_chunks"]} passage(s)</span></div>',
        unsafe_allow_html=True)
    st.markdown("".join(meter(k.title(), r["scores"][k], COLORS[k]) for k in ("positive", "neutral", "negative")),
                unsafe_allow_html=True)
    if r["n_chunks"] > 1:
        st.markdown("**Sentiment through the document** (above 0 is positive, below 0 is negative)")
        st.line_chart(r["trend"])
    c1, c2 = st.columns(2)
    c1.markdown("**Most positive passage**")
    c1.markdown(f"> {_safe(r['best'])}")
    c2.markdown("**Most negative passage**")
    c2.markdown(f"> {_safe(r['worst'])}")
    with st.expander("Passage-by-passage scores"):
        cfg = {k: st.column_config.ProgressColumn(k, min_value=0.0, max_value=1.0, format="%.2f")
               for k in ("Positive", "Neutral", "Negative")}
        st.dataframe(r["table"], column_config=cfg, hide_index=True)
    st.download_button("Download scores (CSV)", r["table"].to_csv(index=False), "sentiment.csv", "text/csv")


def render_ner(r: dict) -> None:
    st.markdown(f'<div class="legend">{r["legend"]}</div>', unsafe_allow_html=True)
    if r["table"].empty:
        st.info("No entities found. Lower the confidence threshold or add more entity types in the sidebar.")
        return
    st.markdown(f'<div class="annot">{r["html"]}</div>', unsafe_allow_html=True)
    if r["clipped"]:
        st.caption(f"Highlighting shows the first {nlp.MAX_HIGHLIGHT_CHUNKS} passages. The table covers everything analysed.")
    c1, c2 = st.columns([3, 2])
    c1.dataframe(r["table"], hide_index=True)
    c2.bar_chart(r["by_type"])
    st.download_button("Download entities (CSV)", r["table"].to_csv(index=False), "entities.csv", "text/csv")


def render_text(r: dict) -> None:
    if r.get("note"):
        st.info(r["note"])
    with st.container(border=True):
        st.markdown(_safe(r["text"]))
    name, data = r["download"]
    st.download_button("Download as text", data, name, "text/plain")


def render_result(r: dict) -> None:
    st.markdown(f"#### {r['title']}")
    {"sentiment": render_sentiment, "ner": render_ner, "text": render_text}[r["kind"]](r)


def execute(task: str, text: str, opts: dict) -> dict:
    if task == "sentiment":
        with st.spinner("Using RoBERT/FinBERT...Did you know: Established in 700 BCE, Takshashila is considered one of the earliest universities in the world, attracting students from across the globe..."):
            return nlp.sentiment(text, opts["financial"])
    if task == "ner":
        with st.spinner("Using GLiNER...Did you know: The Indian Railways network is so massive that it employs over 1.2 million people, which is greater than the entire population of some countries..."):
            return nlp.ner(text, opts["labels"], opts["threshold"])
    if task == "summary":
        with st.spinner("Using Gemini...Did you know: Both Chess (originally called Chaturanga) and Snakes and Ladders originated in ancient India..."):
            return nlp.summarize(text, api_key, opts["length"], opts["style"], opts["focus"])
    if task == "generate":
        with st.spinner("Using GROQ...Did you know: A famous hill in Ladakh has magnetic properties that can pull cars uphill, even when the engine is turned off and the car is in neutral state..."):
            return nlp.generate(text, keys, opts["instruction"], opts["tone"], opts["length"], opts["temperature"])
    bar = st.progress(0.0, text="Using DeepL...Did you know: India is home to the world's only floating post office, which operates out of a traditional houseboat on Dal Lake in Srinagar...")
    try:
        return nlp.translate(text, keys, opts["language"],
                             lambda f: bar.progress(f, text=f"Translating… {int(f * 100)}%"))
    finally:
        bar.empty()


def build_system_prompt() -> str:
    parts = [SYSTEM_BASE]
    doc = ss.doc_text
    if doc.strip():
        note = " (truncated: the document is longer than shown)" if len(doc) > nlp.MAX_DOC_CHARS else ""
        parts.append(f"=== USER'S DOCUMENT: {ss.doc_name}{note} ===\n{doc[:nlp.MAX_DOC_CHARS]}")
    else:
        parts.append("(The user has not loaded a document yet.)")
    if ss.result:
        parts.append(f"=== LATEST NLP OUTPUT: {ss.result['title']} ===\n{ss.result['chat']}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ layout
st.markdown(
    """
    <div class="hero">
      <div>
        <h1>Analyze Research Papers/Projects/Reports</h1>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

main, side = st.columns([5, 3], gap="large")

# ---- left: input + results
with main:
    with st.container(border=True):
        source = st.radio("Input", ["✍️ Paste text", "📎 Upload a file"], horizontal=True, label_visibility="collapsed")
        text, doc_name = "", "Pasted text"
        if source.startswith("✍️"):
            text = st.text_area("Text", height=230, key="pasted", label_visibility="collapsed",
                                placeholder="Paste an article, review, email, contract clause…")
        else:
            up = st.file_uploader("File", type=file_parser.SUPPORTED, label_visibility="collapsed")
            if up is None:
                st.caption("Accepted: PDF, Word (.docx), PowerPoint, Excel, CSV, TXT, Markdown, JSON, HTML.")
            else:
                try:
                    text = file_parser.extract_text(up.name, up.getvalue())
                    doc_name = up.name
                except file_parser.ParseError as e:
                    st.error(str(e))
                else:
                    if not text.strip():
                        st.warning("⚠️ This PDF appears to be a scanned image. Please upload a searchable PDF or paste text directly.")
                    else:
                        st.caption(f"{up.name}: {len(text.split()):,} words, {len(text):,} characters")
                        with st.expander("Preview extracted text"):
                            st.text(text[:3000] + ("…" if len(text) > 3000 else ""))
        ss.doc_text, ss.doc_name = text, doc_name

        run = st.button("Run " + [k for k, v in TASKS.items() if v == task][0].split(" ", 1)[1].lower(), type="primary")
        st.caption(ENGINE_NOTE[task])

    if run:
        ss.result = None
        if not text.strip() and not (task == "generate" and opts["instruction"].strip()):
            st.warning("Paste some text or upload a file first.")
        elif task == "summary" and not keys["gemini"]:
            st.warning("Add your free Gemini API key in the sidebar first.")
        elif task == "generate" and not (keys["groq"] or keys["gemini"]):
            st.warning("Add a free Groq or Gemini API key first (sidebar > Optional keys).")
        else:
            try:
                ss.result = execute(task, text, opts)
            except prov.ProviderError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(gem.friendly_error(e))
    if ss.result:
        render_result(ss.result)

# ---- right: Gemini chat
with side:
    st.markdown('<div class="chat-head"><b>AI Chatbot</b><span>Ask About your document, the results, or anything else</span></div>',
                unsafe_allow_html=True)
    top = st.columns([3, 2])
    with top[0].popover("Suggested questions"):
        for q in QUICK_PROMPTS:
            if st.button(q, key=f"q_{q}"):
                ss.pending = q
    if top[1].button("Clear chat"):
        ss.chat = []

    history = st.container(height=520, border=True)
    prompt = st.chat_input("Type your question") or ss.pop("pending", None)

    with history:
        if not ss.chat and not prompt:
            st.caption("Ask what a passage means, request a simpler explanation, or ask why the sentiment or entities came out this way.")
        for m in ss.chat:
            with st.chat_message(m["role"]):
                st.markdown(_safe(m["content"]))
        if prompt:
            ss.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(_safe(prompt))
            with st.chat_message("assistant"):
                if not api_key:
                    reply = "Add your free Gemini API key in the sidebar to start chatting."
                    st.markdown(reply)
                else:
                    reply = st.write_stream(gem.stream(api_key, ss.chat, build_system_prompt())) or "…"
            ss.chat.append({"role": "assistant", "content": reply})
