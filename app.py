import streamlit as st
import os
import json
import time
import tempfile
from collections import defaultdict
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(
    page_title="MultiMind - Multi-Doc Q&A",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:      #09090f;
    --s1:      #111118;
    --s2:      #18181f;
    --border:  #2a2a3a;
    --acc:     #6c63ff;
    --acc2:    #ff6584;
    --green:   #43e97b;
    --yellow:  #ffd166;
    --text:    #e8e8f5;
    --muted:   #6b6b88;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--border) !important;
}

.hero { padding: 2rem 0 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.hero-title {
    font-size: 3rem; font-weight: 700; letter-spacing: -1.5px;
    background: linear-gradient(135deg, #6c63ff 0%, #ff6584 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; margin: 0;
}
.hero-sub { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; margin-top: 0.4rem; letter-spacing: 0.05em; }

.doc-card {
    background: var(--s2); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.8rem;
    font-size: 0.85rem;
}
.doc-card .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

.stat-card {
    background: var(--s2); border: 1px solid var(--border); border-radius: 10px;
    padding: 1rem; text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: var(--acc); line-height: 1; }
.stat-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }

.qa-card {
    background: var(--s1); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.qa-card.q { border-left: 3px solid var(--acc); }
.qa-card.a { border-left: 3px solid var(--green); background: #0d1a10; }
.qa-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.6rem; }
.qa-text  { font-size: 0.95rem; line-height: 1.8; color: var(--text); }

.source-tag {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    margin: 2px 3px 0 0; border: 1px solid;
}

.conf-high   { background: #0a2a10; color: var(--green);  border-color: #1a4a20; }
.conf-medium { background: #2a2000; color: var(--yellow); border-color: #4a3800; }
.conf-low    { background: #2a0a0a; color: var(--acc2);   border-color: #4a1a1a; }

.chunk-preview {
    background: #050508; border: 1px solid var(--border); border-radius: 8px;
    padding: 0.8rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: var(--muted); line-height: 1.7;
    white-space: pre-wrap; max-height: 150px; overflow-y: auto;
}

.info-box {
    background: var(--s2); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.2rem; margin: 0.8rem 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--muted); line-height: 1.9;
}

.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #ff6584) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.03em !important;
}

.stTextArea textarea {
    background: var(--s2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextArea textarea:focus { border-color: var(--acc) !important; }

.stTextInput input {
    background: var(--s2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stSelectbox > div > div {
    background: var(--s2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1a1a25 !important; border-color: #2a2a3a !important; color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# DOC COLORS for visual distinction
DOC_COLORS = ["#6c63ff","#ff6584","#43e97b","#ffd166","#38bdf8","#fb923c","#a78bfa","#34d399"]

@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

def load_single_doc(uploaded_file):
    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
    pages = loader.load()
    for p in pages:
        p.metadata["source"] = uploaded_file.name
    os.unlink(tmp_path)
    return pages

def build_index(all_docs, chunk_size, chunk_overlap, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". "," ",""],
    )
    chunks = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks

def build_prompt(chunks, question):
    blocks = []
    for i, c in enumerate(chunks):
        src  = c.metadata.get("source","Unknown")
        page = c.metadata.get("page","")
        pi   = (" | Page " + str(int(page)+1)) if page != "" else ""
        blocks.append("SOURCE " + str(i+1) + ": " + src + pi + "\n" + c.page_content)
    context = "\n\n---\n\n".join(blocks)
    return (
        "You are a multi-document Q&A assistant. Answer ONLY from the sources below.\n"
        "Rules:\n"
        "1. Use only the provided sources. No outside knowledge.\n"
        "2. If not found, set answer to: I could not find this in the provided documents.\n"
        "3. Cite which source(s) your answer came from.\n"
        "4. Return valid JSON only with exactly these keys:\n"
        "{\n"
        "  \"answer\": \"your answer\",\n"
        "  \"sources\": [\"file1.pdf\"],\n"
        "  \"confidence\": \"high\" or \"medium\" or \"low\",\n"
        "  \"found_in_documents\": true or false\n"
        "}\n\n"
        "SOURCES:\n\n" + context +
        "\n\nQUESTION: " + question + "\n\nJSON:"
    )

def parse_json(raw):
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"): part = part[4:].strip()
            if part.startswith("{"): text = part; break
    try:
        return json.loads(text)
    except Exception:
        return {"answer": raw, "sources": [], "confidence": "low", "found_in_documents": True}

def ask(api_key, question, vectorstore, top_k, model, temperature):
    chunks = vectorstore.similarity_search(question, k=top_k)
    prompt = build_prompt(chunks, question)
    c = Groq(api_key=api_key)
    r = c.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"Precise multi-document Q&A assistant. Always return valid JSON."},
            {"role":"user","content":prompt},
        ],
        temperature=temperature, max_tokens=1000,
    )
    result = parse_json(r.choices[0].message.content.strip())
    result["chunks"] = chunks
    return result

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:700;'
        'background:linear-gradient(135deg,#6c63ff,#ff6584);-webkit-background-clip:text;'
        '-webkit-text-fill-color:transparent;letter-spacing:-0.5px;">MultiMind</div>',
        unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#6b6b88;margin-bottom:1rem;">Multi-Document RAG Q&A</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Key**")
    st.caption("Free at: https://console.groq.com")
    api_key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")

    st.markdown("**Model**")
    model = st.selectbox("", ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"], label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.markdown("**Chunking**")
    chunk_size    = st.slider("Chunk Size", 200, 1000, 400, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 60, 10)

    st.markdown("**Retrieval**")
    top_k = st.slider("Top K Chunks", 2, 8, 4)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#4a4a60;line-height:1.9;">'
        'Embeddings: all-MiniLM-L6-v2<br>Vector DB: FAISS (local)<br>'
        'Framework: LangChain<br>LLM: Groq (free)<br>Output: JSON + citations'
        '</div>', unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero"><h1 class="hero-title">MultiMind</h1>'
    '<p class="hero-sub">MULTI-DOCUMENT RAG &nbsp;|&nbsp; CROSS-DOC RETRIEVAL &nbsp;|&nbsp; '
    'JSON ANSWERS WITH CITATIONS</p></div>',
    unsafe_allow_html=True)

left, right = st.columns([1,1], gap="large")

# ── LEFT: Upload + Index ──────────────────────────────────────────────────────
with left:
    st.markdown(
        '<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#6c63ff;'
        'text-transform:uppercase;letter-spacing:0.12em;">Step 1 — Upload Documents</p>',
        unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "", type=["pdf","txt"], accept_multiple_files=True,
        label_visibility="collapsed", help="Upload multiple PDF or TXT files")

    if uploaded_files:
        st.markdown('<p style="font-size:0.82rem;color:#6b6b88;margin:0.5rem 0;">Uploaded files:</p>', unsafe_allow_html=True)
        for i, f in enumerate(uploaded_files):
            color = DOC_COLORS[i % len(DOC_COLORS)]
            size  = round(f.size/1024, 1)
            ext   = "PDF" if f.type=="application/pdf" else "TXT"
            st.markdown(
                '<div class="doc-card">'
                '<div class="dot" style="background:' + color + ';"></div>'
                '<span style="flex:1;color:#e8e8f5;">' + f.name + '</span>'
                '<span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#6b6b88;">'
                + ext + ' &nbsp;' + str(size) + 'KB</span>'
                '</div>',
                unsafe_allow_html=True)

        build_btn = st.button("Build Knowledge Base (" + str(len(uploaded_files)) + " files)")

        if build_btn:
            if not api_key:
                st.error("Enter your Groq API key in the sidebar.")
            else:
                with st.spinner("Loading embedding model..."):
                    emb = load_embeddings_model()

                all_pages = []
                with st.spinner("Loading " + str(len(uploaded_files)) + " documents..."):
                    for f in uploaded_files:
                        pages = load_single_doc(f)
                        all_pages.extend(pages)

                with st.spinner("Chunking and indexing..."):
                    vs, chunks = build_index(all_pages, chunk_size, chunk_overlap, emb)

                st.session_state["vs"]       = vs
                st.session_state["chunks"]   = chunks
                st.session_state["doc_names"] = [f.name for f in uploaded_files]
                st.session_state["history"]  = []

                # Count chunks per doc
                cps = defaultdict(int)
                for c in chunks: cps[c.metadata.get("source","?")] += 1
                st.session_state["chunks_per_src"] = dict(cps)

                st.success("Knowledge base ready! " + str(len(chunks)) + " chunks indexed.")

    # Stats
    if "vs" in st.session_state:
        docs_count   = len(st.session_state["doc_names"])
        chunks_count = len(st.session_state["chunks"])
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown('<div class="stat-card"><div class="stat-num">' + str(docs_count) + '</div><div class="stat-lbl">Docs</div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown('<div class="stat-card"><div class="stat-num">' + str(chunks_count) + '</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)
        with sc3:
            st.markdown('<div class="stat-card"><div class="stat-num">' + str(top_k) + '</div><div class="stat-lbl">Top-K</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#6c63ff;text-transform:uppercase;letter-spacing:0.1em;">Chunks per document</p>', unsafe_allow_html=True)
        cps = st.session_state.get("chunks_per_src", {})
        for i, (src, cnt) in enumerate(cps.items()):
            color = DOC_COLORS[i % len(DOC_COLORS)]
            st.markdown(
                '<div class="doc-card">'
                '<div class="dot" style="background:' + color + '"></div>'
                '<span style="flex:1;font-size:0.82rem;color:#e8e8f5;">' + src + '</span>'
                '<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#6b6b88;">'
                + str(cnt) + ' chunks</span></div>',
                unsafe_allow_html=True)

        # Chunk inspector
        with st.expander("Inspect Sample Chunks"):
            shown = {}
            for chunk in st.session_state["chunks"]:
                src = chunk.metadata.get("source","?")
                if src not in shown:
                    shown[src] = chunk
                if len(shown) >= 3: break
            for src, chunk in shown.items():
                st.markdown("**" + src + "** (" + str(len(chunk.page_content)) + " chars)")
                st.markdown('<div class="chunk-preview">' + chunk.page_content[:300] + "...</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:2.5rem;">'
            'Upload multiple PDF or TXT files above<br>'
            '<span style="font-size:0.72rem;opacity:0.5;">then click Build Knowledge Base</span>'
            '</div>', unsafe_allow_html=True)

# ── RIGHT: Q&A ────────────────────────────────────────────────────────────────
with right:
    st.markdown(
        '<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#6c63ff;'
        'text-transform:uppercase;letter-spacing:0.12em;">Step 2 — Ask Questions</p>',
        unsafe_allow_html=True)

    if "vs" not in st.session_state:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:3rem;min-height:300px;">'
            'Build the knowledge base first<br>'
            '<span style="font-size:0.72rem;opacity:0.5;">Upload documents on the left</span>'
            '</div>', unsafe_allow_html=True)
    else:
        question = st.text_area("", height=90, placeholder="Ask anything across your documents...", label_visibility="collapsed")
        ask_btn  = st.button("Ask Question")

        if ask_btn:
            if not question.strip():
                st.warning("Type a question.")
            elif not api_key:
                st.error("Enter your Groq API key.")
            else:
                with st.spinner("Searching across " + str(len(st.session_state["doc_names"])) + " documents..."):
                    try:
                        t0 = time.time()
                        result = ask(api_key, question, st.session_state["vs"], top_k, model, temperature)
                        elapsed = round(time.time()-t0, 2)
                        result["time"] = elapsed
                        result["question"] = question
                        st.session_state["history"].insert(0, result)
                    except Exception as e:
                        err = str(e)
                        if "401" in err or "invalid" in err.lower():
                            st.error("Invalid API key.")
                        elif "429" in err or "rate" in err.lower():
                            st.error("Rate limit hit. Wait a moment.")
                        else:
                            st.error("Error: " + err)

        # History
        if "history" in st.session_state and st.session_state["history"]:
            for item in st.session_state["history"]:
                # Question
                st.markdown(
                    '<div class="qa-card q"><div class="qa-label">Question</div>'
                    '<div class="qa-text">' + item["question"] + '</div></div>',
                    unsafe_allow_html=True)

                # Answer
                st.markdown(
                    '<div class="qa-card a"><div class="qa-label">Answer</div>'
                    '<div class="qa-text">' + item.get("answer","") + '</div></div>',
                    unsafe_allow_html=True)

                # Meta row
                conf  = item.get("confidence","low")
                found = item.get("found_in_documents", True)
                srcs  = item.get("sources", [])
                conf_class = "conf-" + conf

                src_tags = ""
                for i, s in enumerate(srcs):
                    color = DOC_COLORS[
                        st.session_state["doc_names"].index(s)
                        if s in st.session_state["doc_names"] else i
                    ] if srcs else "#6c63ff"
                    src_tags += (
                        '<span class="source-tag" style="background:' + color + '22;'
                        'color:' + color + ';border-color:' + color + '44;">' + s + '</span>'
                    )

                st.markdown(
                    '<div style="margin:0.4rem 0 0.8rem;display:flex;flex-wrap:wrap;align-items:center;gap:6px;">'
                    '<span class="source-tag ' + conf_class + '">Confidence: ' + conf + '</span>'
                    '<span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#6b6b88;">'
                    + str(item.get("time","")) + 's</span>'
                    + src_tags + '</div>',
                    unsafe_allow_html=True)

                # Chunks inspector
                with st.expander("View Retrieved Chunks"):
                    for j, chunk in enumerate(item.get("chunks",[])):
                        src  = chunk.metadata.get("source","?")
                        page = chunk.metadata.get("page","")
                        pi   = " | Page " + str(int(page)+1) if page != "" else ""
                        st.markdown("**Chunk " + str(j+1) + "** — " + src + pi)
                        st.markdown('<div class="chunk-preview">' + chunk.page_content + '</div>', unsafe_allow_html=True)

                st.markdown("---")

            if st.button("Clear History"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.markdown(
                '<div class="info-box" style="text-align:center;min-height:200px;padding:2.5rem;">'
                'Your answers will appear here<br>'
                '<span style="font-size:0.72rem;opacity:0.5;">with source citations and confidence scores</span>'
                '</div>', unsafe_allow_html=True)
