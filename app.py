import streamlit as st
from groq import Groq
import json, time

st.set_page_config(page_title="Text Lens", page_icon="Lens", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
:root{--bg:#0d0d0f;--surface:#16161a;--surface2:#1e1e24;--border:#2a2a35;--accent:#f0b429;--text:#e8e8f0;--muted:#6b6b80;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
.hero-title{font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#f0b429,#e05a2b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero-sub{color:var(--muted);font-size:1rem;margin-top:0.3rem;}
.result-box{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.8rem 2rem;margin-top:1rem;position:relative;overflow:hidden;}
.result-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#f0b429,#e05a2b);}
.rlabel{font-family:'DM Mono',monospace;font-size:0.68rem;color:var(--accent);text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;}
.rtext{font-size:0.97rem;line-height:1.85;color:var(--text);white-space:pre-wrap;}
.jblock{background:#0a0a0f;border:1px solid var(--border);border-radius:10px;padding:1.2rem;font-family:'DM Mono',monospace;font-size:0.82rem;line-height:1.8;color:#a78bfa;white-space:pre-wrap;}
.chip{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:5px 13px;font-family:'DM Mono',monospace;font-size:0.73rem;color:var(--muted);display:inline-block;margin:4px 4px 0 0;}
.chip span{color:var(--text);font-weight:600;}
.stTextArea textarea{background-color:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;}
.stTextArea textarea:focus{border-color:var(--accent)!important;}
.stTextInput input{background-color:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;font-family:'DM Mono',monospace!important;}
.stSelectbox>div>div{background-color:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;}
.stButton>button{background:linear-gradient(135deg,#f0b429,#e05a2b)!important;color:#0d0d0f!important;border:none!important;border-radius:10px!important;font-weight:600!important;width:100%!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;}
</style>
""", unsafe_allow_html=True)

SYSTEM = "You are an expert text summarizer. Be concise, accurate, and follow the output format exactly."

FMT_MAP = {
    "Bullet Points": "bullet points — use the '.' symbol for each point, one per line",
    "Plain Paragraph": "a single coherent paragraph",
    "JSON": 'valid JSON only — keys: "summary"(string), "key_points"(array of strings), "word_count"(int). No markdown fences.',
}

def build_prompt(text, mode, fmt, length):
    fi = FMT_MAP[fmt]
    length_str = str(length)

    if mode == "Zero-Shot":
        return (
            "Summarize the following text in " + length_str + " sentences.\n"
            "Output format: " + fi + "\n\n"
            "Text:\n" + text
        )

    elif mode == "One-Shot":
        return (
            "Here is an example of a good summary:\n\n"
            "INPUT: 'The Apollo 11 mission landed the first humans on the Moon in July 1969. "
            "Neil Armstrong and Buzz Aldrin walked on the surface while Michael Collins orbited above.'\n"
            "SUMMARY:\n"
            ". First crewed Moon landing took place in July 1969.\n"
            ". Neil Armstrong and Buzz Aldrin walked on the lunar surface.\n"
            ". Michael Collins remained in orbit above.\n\n"
            "---\n"
            "Now summarize the following text in " + length_str + " sentences.\n"
            "Output format: " + fi + "\n\n"
            "Text:\n" + text
        )

    else:  # Few-Shot
        return (
            "Study these three examples of high-quality summaries:\n\n"
            "EXAMPLE 1\n"
            "INPUT: 'Climate change refers to long-term shifts in temperatures and weather patterns. "
            "Human activities, particularly fossil fuel burning, have been the main driver since the 1800s.'\n"
            "SUMMARY:\n"
            ". Climate change means long-term shifts in temperature and weather.\n"
            ". Human fossil fuel use since the 1800s is the primary cause.\n\n"
            "EXAMPLE 2\n"
            "INPUT: 'Machine learning is a subset of AI that enables systems to learn from data. "
            "It powers applications from spam filters to self-driving cars.'\n"
            "SUMMARY:\n"
            ". Machine learning allows computers to learn from data automatically.\n"
            ". It powers technologies like spam filters and autonomous vehicles.\n\n"
            "EXAMPLE 3\n"
            "INPUT: 'The Renaissance spanned the 14th to 17th centuries across Europe. "
            "It emphasized humanism, art, and science, marking a shift away from medieval thinking.'\n"
            "SUMMARY:\n"
            ". The Renaissance was a 14th to 17th century European cultural revival.\n"
            ". It promoted humanism, art, and science over medieval traditions.\n\n"
            "---\n"
            "Now summarize the following text in " + length_str + " sentences.\n"
            "Output format: " + fi + "\n\n"
            "Text:\n" + text
        )


def call_groq(api_key, prompt, model, temperature):
    c = Groq(api_key=api_key)
    r = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=800,
    )
    return r.choices[0].message.content.strip()


def fmt_json(raw):
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.dumps(json.loads(clean.strip()), indent=2)
    except Exception:
        lines = [l.strip(". -").strip() for l in raw.splitlines() if l.strip()]
        return json.dumps({
            "summary": lines[0] if lines else raw,
            "key_points": lines[1:],
            "format": "auto-generated"
        }, indent=2)


# Sidebar
with st.sidebar:
    st.markdown(
        '<div style="font-family:Playfair Display,serif;font-size:1.5rem;font-weight:900;'
        'background:linear-gradient(135deg,#f0b429,#e05a2b);-webkit-background-clip:text;'
        '-webkit-text-fill-color:transparent;">TextLens</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div style="color:#6b6b80;font-size:0.88rem;">Groq · 100% Free · No Credit Card</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Key**")
    st.caption("Free key at: https://console.groq.com")
    api_key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")

    st.markdown("**Model**")
    model = st.selectbox("", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ], label_visibility="collapsed")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    st.markdown("**Options**")
    mode = st.selectbox("Prompting Mode", ["Zero-Shot", "One-Shot", "Few-Shot"])
    output_format = st.selectbox("Output Format", ["Bullet Points", "Plain Paragraph", "JSON"])
    summary_length = st.slider("Length (sentences)", 1, 10, 3)

    st.markdown("---")
    model_info = {
        "llama-3.3-70b-versatile": "Best quality · Llama 3.3 70B",
        "llama-3.1-8b-instant":    "Fastest · Llama 3.1 8B",
        "mixtral-8x7b-32768":      "Long text · 32K context",
        "gemma2-9b-it":            "Google Gemma 2 · 9B",
    }
    mode_tips = {
        "Zero-Shot": "Direct instruction. No examples.",
        "One-Shot":  "1 example guides format and tone.",
        "Few-Shot":  "3 examples for max consistency.",
    }
    st.info(model_info.get(model, "") + "\n\n" + mode_tips.get(mode, ""))


# Header
st.markdown(
    '<div style="padding:1.5rem 0 1rem;">'
    '<h1 class="hero-title">TextLens</h1>'
    '<p class="hero-sub">Summarize any article with Zero, One, or Few-Shot prompting · Powered by Groq (Free)</p>'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.7rem;color:#f0b429;'
        'letter-spacing:0.1em;text-transform:uppercase;">Input Text</p>',
        unsafe_allow_html=True,
    )
    text = st.text_area(
        "",
        height=300,
        placeholder="Paste your article, news story, or any long text here...",
        label_visibility="collapsed",
    )
    wc = len(text.split()) if text.strip() else 0
    st.markdown(
        '<div>'
        '<span class="chip">Words <span>' + str(wc) + '</span></span>'
        '<span class="chip">Mode <span>' + mode + '</span></span>'
        '<span class="chip">Format <span>' + output_format + '</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    go = st.button("Summarize")

with col2:
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.7rem;color:#f0b429;'
        'letter-spacing:0.1em;text-transform:uppercase;">Summary Output</p>',
        unsafe_allow_html=True,
    )

    if go:
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        elif not text.strip():
            st.error("Paste some text to summarize.")
        elif wc < 15:
            st.warning("Text too short — add more content.")
        else:
            prompt = build_prompt(text, mode, output_format, summary_length)
            with st.spinner("Summarizing with " + model + "..."):
                try:
                    t0 = time.time()
                    raw = call_groq(api_key, prompt, model, temperature)
                    elapsed = round(time.time() - t0, 2)
                    result = fmt_json(raw) if output_format == "JSON" else raw
                    st.session_state.update({
                        "r": result,
                        "t": elapsed,
                        "wi": wc,
                        "wo": len(result.split()),
                        "fmt": output_format,
                    })
                except Exception as e:
                    err = str(e)
                    if "401" in err or "api_key" in err.lower() or "invalid" in err.lower():
                        st.error("Invalid API key. Check your key at console.groq.com")
                    elif "429" in err or "rate" in err.lower():
                        st.error("Rate limit hit. Wait 10 seconds and retry.")
                    else:
                        st.error("Error: " + err)

    if "r" in st.session_state:
        r   = st.session_state["r"]
        fmt = st.session_state["fmt"]
        body = '<div class="jblock">' + r + '</div>' if fmt == "JSON" else '<div class="rtext">' + r + '</div>'
        st.markdown(
            '<div class="result-box"><div class="rlabel">Output — ' + fmt + '</div>' + body + '</div>',
            unsafe_allow_html=True,
        )
        wi   = st.session_state["wi"]
        wo   = st.session_state["wo"]
        comp = round((1 - wo / max(wi, 1)) * 100)
        st.markdown(
            '<div style="margin-top:.8rem;">'
            '<span class="chip">Time <span>' + str(st.session_state["t"]) + 's</span></span>'
            '<span class="chip">In <span>' + str(wi) + 'w</span></span>'
            '<span class="chip">Out <span>' + str(wo) + 'w</span></span>'
            '<span class="chip">Compressed <span>' + str(comp) + '%</span></span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        ext = "json" if fmt == "JSON" else "txt"
        st.download_button(
            "Download Summary",
            r,
            file_name="summary." + ext,
            mime="application/json" if fmt == "JSON" else "text/plain",
        )
    else:
        st.markdown(
            '<div class="result-box" style="min-height:280px;display:flex;align-items:center;'
            'justify-content:center;flex-direction:column;color:#2a2a35;">'
            '<div style="font-size:2.5rem;">*</div>'
            '<div style="font-family:DM Mono,monospace;font-size:0.75rem;letter-spacing:0.1em;margin-top:.6rem;">'
            'SUMMARY WILL APPEAR HERE</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# Prompt Inspector
if text.strip():
    with st.expander("Inspect Prompt Sent to Model"):
        preview = build_prompt(
            text[:300] + ("..." if len(text) > 300 else ""),
            mode,
            output_format,
            summary_length,
        )
        st.code(preview, language="text")
