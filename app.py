import streamlit as st
import tempfile
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
    }
    .verdict-high {
        background: linear-gradient(135deg, #ff416c22, #ff4b2b22);
        border: 1px solid #ff416c55;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .verdict-medium {
        background: linear-gradient(135deg, #f7971e22, #ffd20022);
        border: 1px solid #f7971e55;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .verdict-low {
        background: linear-gradient(135deg, #11998e22, #38ef7d22);
        border: 1px solid #11998e55;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        margin: 1.5rem 0 0.8rem 0;
        border-bottom: 1px solid #313244;
        padding-bottom: 0.4rem;
    }
    .weight-badge {
        display: inline-block;
        background: #313244;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.85rem;
        margin: 0.2rem;
        font-family: monospace;
    }
    .stProgress > div > div > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Lazy-load backend (cached so model loads only once) ───────────────────────
@st.cache_resource(show_spinner="Loading SPECTER model — this takes ~30s on first run...")
def load_backend():
    """Import pipeline_max functions after model is cached."""
    import pipeline_max as pm
    return pm


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp path and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return f.name


def score_color(value: float) -> str:
    if value >= 0.65:
        return "#ff416c"
    elif value >= 0.45:
        return "#f7971e"
    return "#11998e"


def make_gauge(value: float, title: str) -> go.Figure:
    color = score_color(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": title, "font": {"size": 15, "color": "#cdd6f4"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#585b70",
                     "tickfont": {"color": "#585b70"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e1e2e",
            "bordercolor": "#313244",
            "steps": [
                {"range": [0, 45],  "color": "rgba(17,153,142,0.13)"},
                {"range": [45, 65], "color": "rgba(247,151,30,0.13)"},
                {"range": [65, 100],"color": "rgba(255,65,108,0.13)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#181825",
        font_color="#cdd6f4",
    )
    return fig


def make_radar(results: dict, weights: dict) -> go.Figure:
    labels = ["S_sem", "S_struct", "S_align", "S_cite", "S_common"]
    values = [
        results["S_sem"],
        results["S_struct"],
        results["S_align"],
        results["S_cite"],
        results["S_common"],
    ]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(102,126,234,0.2)",
        line=dict(color="#667eea", width=2),
        marker=dict(size=6, color="#667eea"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e1e2e",
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(color="#585b70"), gridcolor="#313244"),
            angularaxis=dict(tickfont=dict(color="#cdd6f4"), gridcolor="#313244"),
        ),
        paper_bgcolor="#181825",
        font_color="#cdd6f4",
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
        showlegend=False,
    )
    return fig


def make_bar(results: dict) -> go.Figure:
    components = {
        "S_sem\n(Semantic)":    results["S_sem"],
        "S_struct\n(Structure)": results["S_struct"],
        "S_align\n(N-gram)":    results["S_align"],
        "S_cite\n(Citation)":   results["S_cite"],
        "S_common\n(Common)":   results["S_common"],
    }
    colors = [score_color(v) for v in components.values()]
    fig = go.Figure(go.Bar(
        x=list(components.keys()),
        y=list(components.values()),
        marker_color=colors,
        marker_line_color="#313244",
        marker_line_width=1,
        text=[f"{v:.3f}" for v in components.values()],
        textposition="outside",
        textfont=dict(color="#cdd6f4", size=12),
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1.15], gridcolor="#313244",
                   tickfont=dict(color="#585b70"), title="Score"),
        xaxis=dict(tickfont=dict(color="#cdd6f4")),
        paper_bgcolor="#181825",
        plot_bgcolor="#181825",
        font_color="#cdd6f4",
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Mode")
    input_mode = st.radio(
        "Choose input type",
        ["📄 Upload PDFs", "✏️ Paste Raw Text"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📖 Score Guide")
    st.markdown("""
| Range | Verdict |
|-------|---------|
| ≥ 0.65 | 🔴 HIGH RISK |
| 0.45 – 0.65 | 🟡 MEDIUM RISK |
| < 0.45 | 🟢 LOW RISK |
""")
    st.markdown("---")
    st.markdown("### 🧮 Formula")
    st.latex(r"R = \alpha S_{sem} + \beta S_{struct} + \gamma S_{align} + \delta S_{cite} - \epsilon S_{common}")
    st.markdown("""
- **α = 0.45** Semantic similarity  
- **β = 0.10** Structural consistency  
- **γ = 0.25** N-gram alignment  
- **δ = 0.15** Citation score  
- **ε = 0.05** Common knowledge penalty  
""")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 Academic Plagiarism Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by SPECTER embeddings · N-gram alignment · Citation analysis</div>', unsafe_allow_html=True)

# ── Input Section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

doc_a_text = doc_b_text = None
pdf_a_path = pdf_b_path = None

if input_mode == "📄 Upload PDFs":
    with col1:
        st.markdown("#### 📄 Document A (Original / Reference)")
        pdf_a = st.file_uploader("Upload PDF A", type=["pdf"], key="pdf_a",
                                  label_visibility="collapsed")
        if pdf_a:
            st.success(f"✓ {pdf_a.name}  ({pdf_a.size // 1024} KB)")

    with col2:
        st.markdown("#### 📄 Document B (Suspect)")
        pdf_b = st.file_uploader("Upload PDF B", type=["pdf"], key="pdf_b",
                                  label_visibility="collapsed")
        if pdf_b:
            st.success(f"✓ {pdf_b.name}  ({pdf_b.size // 1024} KB)")

    ready = pdf_a is not None and pdf_b is not None

else:  # Raw text mode
    with col1:
        st.markdown("#### ✏️ Document A (Original / Reference)")
        doc_a_text = st.text_area(
            "Paste Document A",
            height=220,
            placeholder="Paste the original or reference document here...",
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("#### ✏️ Document B (Suspect)")
        doc_b_text = st.text_area(
            "Paste Document B",
            height=220,
            placeholder="Paste the suspect document here...",
            label_visibility="collapsed",
        )

    ready = bool(doc_a_text and doc_a_text.strip() and doc_b_text and doc_b_text.strip())

st.markdown("")
run_btn = st.button("🚀 Run Analysis", type="primary", disabled=not ready, use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn and ready:
    pm = load_backend()

    with st.spinner("Running plagiarism detection pipeline..."):
        progress = st.progress(0, text="Preprocessing documents...")

        # --- Prepare text ---
        if input_mode == "📄 Upload PDFs":
            path_a = save_upload(pdf_a)
            path_b = save_upload(pdf_b)
            progress.progress(10, text="Extracting text from PDFs...")
            text_a = pm.extract_pdf_text(path_a)
            text_b = pm.extract_pdf_text(path_b)
            os.unlink(path_a)
            os.unlink(path_b)
        else:
            progress.progress(10, text="Preprocessing text...")
            text_a = pm.preprocess_with_c(doc_a_text)
            text_b = pm.preprocess_with_c(doc_b_text)

        progress.progress(25, text="Computing semantic embeddings (S_sem)...")
        # Run full pipeline
        results = pm.calculate_pipeline(text_a, text_b)
        progress.progress(100, text="Done!")
        progress.empty()

    # ── Results ───────────────────────────────────────────────────────────────
    r = results["Risk Score (R)"]
    weights = results["Final Weights"]

    st.markdown("---")

    # Verdict banner
    if r >= 0.65:
        verdict_class = "verdict-high"
        verdict_icon  = "🔴"
        verdict_text  = "HIGH RISK — Likely Plagiarism"
        verdict_color = "#ff416c"
    elif r >= 0.45:
        verdict_class = "verdict-medium"
        verdict_icon  = "🟡"
        verdict_text  = "MEDIUM RISK — Review Recommended"
        verdict_color = "#f7971e"
    else:
        verdict_class = "verdict-low"
        verdict_icon  = "🟢"
        verdict_text  = "LOW RISK — Likely Original"
        verdict_color = "#11998e"

    st.markdown(f"""
    <div class="{verdict_class}">
        <div style="font-size:2rem;">{verdict_icon}</div>
        <div style="font-size:1.6rem; font-weight:700; color:{verdict_color}; margin:0.3rem 0;">
            {verdict_text}
        </div>
        <div style="font-size:1rem; color:#888;">
            Risk Score: <strong style="color:{verdict_color}; font-size:1.3rem;">{r:.4f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Gauge + Radar ─────────────────────────────────────────────────────────
    g_col, r_col = st.columns([1, 1.4])

    with g_col:
        st.markdown('<div class="section-header">Risk Score Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(r, "Overall Risk Score"), use_container_width=True)

        # Citations found
        cite_count = results["Citations Found"]
        st.metric(
            label="📚 Citations Found in Doc A",
            value=cite_count,
            delta="Weight redistributed" if cite_count == 0 else f"δ = {weights['DELTA']}",
            delta_color="off" if cite_count == 0 else "normal",
        )

    with r_col:
        st.markdown('<div class="section-header">Component Score Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar(results, weights), use_container_width=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Component Breakdown</div>', unsafe_allow_html=True)
    st.plotly_chart(make_bar(results), use_container_width=True)

    # ── Score cards ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Detailed Scores</div>', unsafe_allow_html=True)

    score_meta = [
        ("S_sem",    "🧠 Semantic Similarity",      "SPECTER [CLS] cosine similarity",          "#667eea"),
        ("S_struct", "🏗️ Structural Consistency",   "Paragraph-level embedding alignment",       "#a6e3a1"),
        ("S_align",  "🔗 N-gram Alignment",          "6-gram Jaccard overlap (verbatim detector)","#fab387"),
        ("S_cite",   "📖 Citation Score",            "Risk from low citations vs high similarity","#f38ba8"),
        ("S_common", "📚 Common Knowledge Penalty",  "Boilerplate phrase similarity (subtracted)","#89dceb"),
    ]

    c1, c2 = st.columns(2)
    for idx, (key, label, desc, color) in enumerate(score_meta):
        val = results[key]
        col = c1 if idx % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color};">
                <div style="font-size:0.8rem; color:#888; margin-bottom:0.2rem;">{label}</div>
                <div style="font-size:1.6rem; font-weight:700; color:{color};">{val:.4f}</div>
                <div style="font-size:0.75rem; color:#585b70; margin-top:0.2rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(np.clip(val, 0, 1)))

    # ── Final weights ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Final Weights Used</div>', unsafe_allow_html=True)
    w_cols = st.columns(5)
    weight_labels = {
        "ALPHA":   ("α", "Semantic"),
        "BETA":    ("β", "Structure"),
        "GAMMA":   ("γ", "N-gram"),
        "DELTA":   ("δ", "Citation"),
        "EPSILON": ("ε", "Common"),
    }
    for col, (wk, (sym, name)) in zip(w_cols, weight_labels.items()):
        with col:
            st.metric(label=f"{sym} {name}", value=weights[wk])

    # ── Formula expansion ─────────────────────────────────────────────────────
    with st.expander("🧮 Show full score calculation"):
        a = float(weights["ALPHA"])
        b = float(weights["BETA"])
        g = float(weights["GAMMA"])
        d = float(weights["DELTA"])
        e = float(weights["EPSILON"])
        s = results["S_sem"]
        st2 = results["S_struct"]
        al = results["S_align"]
        ci = results["S_cite"]
        co = results["S_common"]

        st.code(f"""
R = (α × S_sem)   + (β × S_struct) + (γ × S_align) + (δ × S_cite) - (ε × S_common)
  = ({a:.3f} × {s:.4f}) + ({b:.3f} × {st2:.4f}) + ({g:.3f} × {al:.4f}) + ({d:.3f} × {ci:.4f}) - ({e:.3f} × {co:.4f})
  = {a*s:.4f}        + {b*st2:.4f}        + {g*al:.4f}        + {d*ci:.4f}        - {e*co:.4f}
  = {r:.4f}
        """, language="text")

    # ── Preprocessed text preview ─────────────────────────────────────────────
    with st.expander("📝 View preprocessed text"):
        ta, tb = st.tabs(["Document A", "Document B"])
        with ta:
            st.text_area("doc_a_preview", value=text_a[:3000] + ("..." if len(text_a) > 3000 else ""),
                         height=200, disabled=True, label_visibility="collapsed", key="preview_a")
        with tb:
            st.text_area("doc_b_preview", value=text_b[:3000] + ("..." if len(text_b) > 3000 else ""),
                         height=200, disabled=True, label_visibility="collapsed", key="preview_b")

# ── Empty state ───────────────────────────────────────────────────────────────
elif not run_btn:
    st.markdown("")
    st.info("👆 Upload two PDFs or paste text in both boxes, then click **Run Analysis**.")
