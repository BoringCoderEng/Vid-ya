import os
import time
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# ============================================================================
# Pipeline (same logic as your CLI script — just instrumented for the UI)
# ============================================================================
PIPELINE_STEPS = [
    ("🎧", "Pulling audio out of your file..."),
    ("📝", "Transcribing every word..."),
    ("🏷️", "Coming up with a title..."),
    ("🧠", "Summarizing the good stuff..."),
    ("✅", "Hunting for action items..."),
    ("🔑", "Spotting key decisions..."),
    ("❓", "Collecting open questions..."),
    ("🔗", "Wiring up the chat brain..."),
]


def run_pipeline(source: str, language: str = "english", progress_cb=None) -> dict:
    def step(i):
        if progress_cb:
            progress_cb(i)

    step(0)
    chunks = process_input(source)

    step(1)
    transcript = transcribe_all(chunks, language=language)
    print(f"raw transcription: {transcript[:300]}")

    step(2)
    title = generate_title(transcript)

    step(3)
    summary = summarize(transcript)

    step(4)
    action_items = extract_action_items(transcript)

    step(5)
    decisions = extract_key_decisions(transcript)

    step(6)
    questions = extract_questions(transcript)

    step(7)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decision": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "language": language,
        "processed_at": datetime.now().strftime("%b %d, %Y — %I:%M %p"),
    }


# ============================================================================
# Helpers
# ============================================================================
def normalize_to_list(data):
    """Make action items / decisions / questions render nicely whether the
    underlying function returns a list, dict, or a raw string."""
    if data is None:
        return []
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    if isinstance(data, dict):
        return [f"{k}: {v}" for k, v in data.items()]
    if isinstance(data, str):
        parts = [p.strip(" -•*\t") for p in data.split("\n") if p.strip(" -•*\t")]
        return parts if parts else [data.strip()]
    return [str(data)]


def render_checklist(items, empty_msg, key_prefix):
    items = normalize_to_list(items)
    if not items:
        st.info(empty_msg)
        return
    for i, item in enumerate(items):
        st.checkbox(item, key=f"{key_prefix}_{i}")


def build_report_markdown(result: dict) -> str:
    def bullets(data):
        return "\n".join(f"- {x}" for x in normalize_to_list(data)) or "_None found_"

    return f"""# {result['title']}
_Processed {result['processed_at']} • Language: {result['language']}_

## Summary
{result['summary']}

## Action Items
{bullets(result['action_items'])}

## Key Decisions
{bullets(result['key_decision'])}

## Open Questions
{bullets(result['open_questions'])}

## Full Transcript
{result['transcript']}
"""


QUICK_QUESTIONS = [
    "Give me a 2-line TL;DR",
    "What are the biggest risks mentioned?",
    "Who is responsible for what?",
    "What's still unresolved?",
]


# ============================================================================
# Page setup + styling
# ============================================================================
st.set_page_config(page_title="AI Video Assistant", page_icon="🎥", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #6a5cf7 0%, #b45cf7 50%, #f75c9e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle { color: #8a8a9a; font-size: 1.05rem; margin-top: -8px; }
    div[data-testid="stMetric"] {
        background: rgba(120, 100, 250, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid rgba(120, 100, 250, 0.18);
    }
    .stChatMessage { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🎥 AI Video Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Drop in a video or meeting — get a transcript, summary, '
    'action items, and a chat buddy who actually watched it.</p>',
    unsafe_allow_html=True,
)
st.write("")

# ============================================================================
# Session state
# ============================================================================
st.session_state.setdefault("result", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("processing", False)
st.session_state.setdefault("history", [])  # list of past results, most recent first
st.session_state.setdefault("pending_question", None)

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("⚙️ New Video")

    source_type = st.radio("Source", ["🔗 YouTube URL", "📁 Upload file"], horizontal=False)

    source = None
    if source_type == "🔗 YouTube URL":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if url:
            source = url
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video",
            type=["mp4", "mp3", "wav", "m4a", "mov", "webm"],
        )
        if uploaded_file is not None:
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = tmp_path
            st.success(f"📎 {uploaded_file.name} ready")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    process_clicked = st.button(
        "🚀 Process video",
        type="primary",
        disabled=(source is None or st.session_state.processing),
        use_container_width=True,
    )

    if st.session_state.result is not None:
        if st.button("✨ Process another", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.history:
        st.divider()
        st.header("🕘 History")
        for i, past in enumerate(st.session_state.history):
            label = f"{past['title'][:30]}{'…' if len(past['title']) > 30 else ''}"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.result = past
                st.session_state.chat_history = []
                st.rerun()

# ============================================================================
# Run pipeline with a lively progress bar
# ============================================================================
if process_clicked and source:
    st.session_state.processing = True
    status_box = st.empty()
    progress_bar = st.progress(0.0)
    start_time = time.time()

    def progress_cb(step_index):
        emoji, msg = PIPELINE_STEPS[step_index]
        pct = (step_index) / len(PIPELINE_STEPS)
        progress_bar.progress(pct)
        status_box.info(f"{emoji} {msg}")

    try:
        result = run_pipeline(source, language=language, progress_cb=progress_cb)
        elapsed = time.time() - start_time
        result["elapsed_seconds"] = round(elapsed, 1)

        st.session_state.result = result
        st.session_state.chat_history = []
        st.session_state.history.insert(0, result)
        st.session_state.history = st.session_state.history[:10]

        progress_bar.progress(1.0)
        status_box.success(f"🎉 All done in {elapsed:.1f}s!")
        st.balloons()
    except Exception as e:
        status_box.error(f"❌ Something broke: {e}")
        with st.expander("Show error details"):
            st.exception(e)
    finally:
        st.session_state.processing = False

# ============================================================================
# Results
# ============================================================================
result = st.session_state.result

if result is None:
    st.info("👈 Add a YouTube URL or upload a file in the sidebar, then hit **Process video**.")
    st.write("")
    st.markdown("##### What you'll get:")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("📝 **Full transcript**")
    c2.markdown("🧠 **Smart summary**")
    c3.markdown("✅ **Action items**")
    c4.markdown("💬 **Chat with it**")
else:
    st.header(f"📌 {result['title']}")
    st.caption(f"Processed {result.get('processed_at', '')} · Language: {result.get('language', 'english')}")

    # ---- stat row ----
    a_count = len(normalize_to_list(result["action_items"]))
    d_count = len(normalize_to_list(result["key_decision"]))
    q_count = len(normalize_to_list(result["open_questions"]))
    word_count = len(result["transcript"].split())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📝 Words transcribed", f"{word_count:,}")
    m2.metric("✅ Action items", a_count)
    m3.metric("🔑 Decisions", d_count)
    m4.metric("❓ Open questions", q_count)

    st.write("")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs(
        ["📋 Summary", f"✅ Action Items ({a_count})", f"🔑 Decisions ({d_count})",
         f"❓ Questions ({q_count})", "📄 Transcript"]
    )

    with tab_summary:
        st.write(result["summary"])

    with tab_actions:
        render_checklist(result["action_items"], "No action items detected 🎉", "action")

    with tab_decisions:
        items = normalize_to_list(result["key_decision"])
        if not items:
            st.info("No clear decisions detected.")
        for item in items:
            st.markdown(f"> 🔑 {item}")

    with tab_questions:
        items = normalize_to_list(result["open_questions"])
        if not items:
            st.info("No open questions left dangling — nice and tidy!")
        for item in items:
            st.markdown(f"❓ {item}")

    with tab_transcript:
        st.text_area("Full transcript", result["transcript"], height=350, label_visibility="collapsed")

    st.write("")
    dl1, dl2, _ = st.columns([1, 1, 3])
    with dl1:
        st.download_button(
            "⬇️ Download full report (.md)",
            data=build_report_markdown(result),
            file_name=f"{result['title'][:40].replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["transcript"],
            file_name=f"{result['title'][:40].replace(' ', '_')}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # Chat with the meeting (RAG)
    # ------------------------------------------------------------------
    st.subheader("💬 Chat with your meeting")

    chip_cols = st.columns(len(QUICK_QUESTIONS))
    for i, q in enumerate(QUICK_QUESTIONS):
        if chip_cols[i].button(q, key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_question = q

    for role, msg in st.session_state.chat_history:
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.write(msg)

    typed_question = st.chat_input("Ask anything about this video...")
    question = st.session_state.pending_question or typed_question
    st.session_state.pending_question = None

    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user", avatar="🧑"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Digging through the transcript..."):
                try:
                    answer = ask_question(result["rag_chain"], question)
                except Exception as e:
                    answer = f"❌ Error answering question: {e}"
            st.write(answer)

        st.session_state.chat_history.append(("assistant", answer))

    if st.session_state.chat_history:
        if st.button("🧹 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()