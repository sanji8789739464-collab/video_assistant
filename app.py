"""
AI Video Assistant — Streamlit UI
Premium Dark Glassmorphism Theme.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_audio_file
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Assistant",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Root variables ──────────────────────────────────────────────── */
:root {
    --bg:       #0B0E14;
    --bg-alt:   #151A22;
    --surface:  rgba(21, 26, 34, 0.6);
    --border:   rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.2);
    --text:     #F3F4F6;
    --text-sec: #9CA3AF;
    --accent-1: #38BDF8;
    --accent-2: #818CF8;
    --accent-grad: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    --radius:   16px;
}

/* ── Global resets ───────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    background-image: radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.04), transparent 25%),
                      radial-gradient(circle at 85% 30%, rgba(129, 140, 248, 0.04), transparent 25%) !important;
    color: var(--text) !important;
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Typography ──────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

p, li, span, label, div {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Hero header ─────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 4rem 1rem 2rem;
    animation: fadeDown 0.8s ease forwards;
}
.hero h1 {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em;
    margin-bottom: 0.5rem;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.hero p {
    color: var(--text-sec);
    font-size: 1.15rem;
    font-weight: 400;
    margin: 0;
}

/* ── Divider line ────────────────────────────────────────────────── */
.divider {
    width: 64px;
    height: 3px;
    background: var(--accent-grad);
    margin: 2rem auto 3rem;
    border-radius: 2px;
    opacity: 0.8;
}

/* ── Stat pills ──────────────────────────────────────────────────── */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin: 1.5rem 0 2rem;
}
.stat-pill {
    background: var(--surface);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.6rem 1.6rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: 0.02em;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.stat-pill:hover {
    transform: translateY(-2px);
    border-color: var(--border-hover);
}

/* ── Card ────────────────────────────────────────────────────────── */
.card {
    background: var(--surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
}
.card-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-1);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.card-label .icon {
    font-size: 1rem;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card-body {
    font-size: 1rem;
    line-height: 1.8;
    color: var(--text);
}
.card-body ul {
    padding-left: 1.2rem;
    margin: 0;
}
.card-body li {
    margin-bottom: 0.5rem;
}
.card-body li::marker {
    color: var(--accent-2);
}

/* ── Chat area ───────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.5rem !important;
    margin-bottom: 0.8rem !important;
    font-size: 0.95rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* assistant bubble slightly different */
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: rgba(56, 189, 248, 0.05) !important;
    border-color: rgba(56, 189, 248, 0.15) !important;
}

/* Chat Avatars */
[data-testid="stChatMessageAvatar"] {
    background-color: var(--bg-alt) !important;
}
[data-testid="stChatMessageAvatar"] svg {
    fill: var(--text) !important;
    color: var(--text) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: var(--bg-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 100px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem;
    padding: 0.8rem 1.5rem !important;
    color: var(--text) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 1px var(--accent-1) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--accent-grad) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 0.65rem 2.2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.3) !important;
    opacity: 0.95;
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Text input ──────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: var(--bg-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem;
    padding: 0.8rem 1.2rem !important;
    color: var(--text) !important;
    transition: all 0.25s ease;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 1px var(--accent-1) !important;
}
[data-testid="stTextInput"] label {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--text-sec) !important;
    margin-bottom: 0.4rem !important;
}

/* ── Select box ──────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-baseweb="select"] svg {
    fill: var(--text-sec) !important;
    color: var(--text-sec) !important;
}
/* Selectbox dropdown menu (BaseWeb popover) */
[data-baseweb="menu"], [data-baseweb="popover"] > div {
    background-color: var(--bg-alt) !important;
}
li[role="option"] {
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background-color: var(--surface) !important;
}

/* ── Expander / Transcript ───────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    backdrop-filter: blur(12px);
}
[data-testid="stExpander"] > summary {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary svg {
    fill: var(--text-sec) !important;
    color: var(--text-sec) !important;
}

/* ── Progress / Spinner ──────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--accent-1) !important;
}
[data-testid="stStatusWidget"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: var(--radius) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stStatusWidget"] label {
    color: var(--text) !important;
}

/* ── Tabs (used for results) ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    color: var(--text-sec);
    padding: 0.8rem 1rem;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    transition: color 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text);
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--accent-1) !important;
}

/* ── Animations ──────────────────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-sec); }

/* ── Hide Streamlit branding ─────────────────────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State ────────────────────────────────────────────────────────────
defaults = {
    "result": None,
    "rag_chain": None,
    "messages": [],
    "processing": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helper: render a card ───────────────────────────────────────────────────
def render_card(icon: str, label: str, content: str):
    st.markdown(
        f"""
        <div class="card animate-in">
            <div class="card-label"><span class="icon">{icon}</span> {label}</div>
            <div class="card-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def md_to_html_basic(md_text: str) -> str:
    """Very lightweight markdown → HTML for bullet lists."""
    if not md_text:
        return ""
    lines = md_text.strip().split("\n")
    html_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "• ", "* ")):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped[2:]}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if stripped:
                html_lines.append(f"<p>{stripped}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Video Assistant</h1>
        <p>Extract insights from any meeting — powered by AI</p>
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True,
)


# ─── Input Section ───────────────────────────────────────────────────────────
if st.session_state.result is None:
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        source = st.text_input(
            "Source",
            placeholder="Paste a YouTube URL",
            label_visibility="collapsed",
        )
        uploaded_file = st.file_uploader("Or upload an audio/video file directly to bypass YouTube blocks", type=["mp4", "mp3", "wav", "m4a", "webm", "mov"])

        lang_col, btn_col = st.columns([1, 1])
        with lang_col:
            language = st.selectbox(
                "Language",
                ["english", "hinglish"],
                index=0,
                label_visibility="collapsed",
            )
        with btn_col:
            process_btn = st.button("Process →", use_container_width=True)

        # ── Run the pipeline ──
        if process_btn and (source.strip() or uploaded_file):
            st.session_state.processing = True

            progress = st.empty()

            with st.status("Processing your video…", expanded=True) as status:
                st.write("⏳  Extracting audio…")
                if uploaded_file:
                    import os
                    os.makedirs("downloads", exist_ok=True)
                    temp_path = os.path.join("downloads", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    chunks = process_audio_file(temp_path)
                else:
                    chunks = process_audio_file(source.strip())

                st.write("⏳  Transcribing audio…")
                transcript = transcribe_all(chunks, language)

                st.write("⏳  Generating title…")
                title = generate_title(transcript)

                st.write("⏳  Summarising…")
                summary_text = summarize(transcript)

                st.write("⏳  Extracting action items…")
                action_items = extract_action_items(transcript)

                st.write("⏳  Extracting key decisions…")
                decisions = extract_key_decisions(transcript)

                st.write("⏳  Extracting open questions…")
                questions = extract_questions(transcript)

                st.write("⏳  Building knowledge base…")
                rag = build_rag_chain(transcript)

                status.update(label="✓  Processing complete", state="complete")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary_text,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
            }
            st.session_state.rag_chain = rag
            st.session_state.processing = False
            st.rerun()

        elif process_btn and not (source.strip() or uploaded_file):
            st.warning("Please enter a YouTube URL or upload a file.")


# ─── Results ─────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    r = st.session_state.result

    # ── Title ──
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 0.5rem;">
            <h2 style="font-size:1.8rem; margin-bottom:0.2rem; background: var(--accent-grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block;">{r["title"]}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick stats ──
    word_count = len(r["transcript"].split()) if r["transcript"] else 0
    action_count = r["action_items"].strip().count("\n") + 1 if r["action_items"] and r["action_items"].strip() else 0
    decision_count = r["key_decisions"].strip().count("\n") + 1 if r["key_decisions"] and r["key_decisions"].strip() else 0
    st.markdown(
        f"""
        <div class="stat-row">
            <span class="stat-pill">~{word_count:,} words transcribed</span>
            <span class="stat-pill">{action_count} action items</span>
            <span class="stat-pill">{decision_count} decisions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabbed results ──
    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["Summary", "Action Items", "Decisions", "Questions", "Transcript", "Chat"]
    )

    with tab_summary:
        render_card("◉", "MEETING SUMMARY", md_to_html_basic(r["summary"]))

    with tab_actions:
        render_card("✓", "ACTION ITEMS", md_to_html_basic(r["action_items"]))

    with tab_decisions:
        render_card("◆", "KEY DECISIONS", md_to_html_basic(r["key_decisions"]))

    with tab_questions:
        render_card("?", "OPEN QUESTIONS", md_to_html_basic(r["open_questions"]))

    with tab_transcript:
        with st.expander("View full transcript", expanded=False):
            st.text(r["transcript"])

    # ── Chat Tab ──
    with tab_chat:
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:1.5rem;">
                <p style="color: var(--text-sec); font-size:0.95rem;">
                    Ask anything about the meeting — the AI will answer from the transcript.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display existing messages
        for msg in st.session_state.messages:
            avatar = "👤" if msg["role"] == "user" else "✨"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the meeting…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="✨"):
                with st.spinner(""):
                    try:
                        from core.rag_engine import load_rag_chain
                        chain = load_rag_chain()
                        answer = ask_question(chain, prompt)
                    except Exception as e:
                        answer = f"Oops, something went wrong querying the document: {e}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── New session button ──
    st.markdown("<br>", unsafe_allow_html=True)
    _, reset_col, _ = st.columns([2, 1, 2])
    with reset_col:
        if st.button("New Session ↻", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()
