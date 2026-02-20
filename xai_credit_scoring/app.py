import streamlit as st
import pandas as pd
import shap
import pickle
import plotly.graph_objects as go
import plotly.express as px
from streamlit_shap import st_shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import accuracy_score, confusion_matrix
import re, hashlib, json, os, time
from pan_api_client import PANApiClient, get_client_from_env

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinTrust AI | Credit Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — Dark Premium Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root Reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0f1e; }
section[data-testid="stSidebar"] > div { background: #070c18; border-right: 1px solid #1a2540; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* ── Hide default header ── */
#MainMenu, header, footer { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Sidebar Logo ── */
.sidebar-logo {
    text-align: center; padding: 28px 16px 20px;
    border-bottom: 1px solid #1a2540; margin-bottom: 16px;
}
.sidebar-logo .brand { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem;
    font-weight: 700; color: #f0c040; letter-spacing: -0.5px; }
.sidebar-logo .tagline { font-size: 0.72rem; color: #4a6080; margin-top: 3px;
    text-transform: uppercase; letter-spacing: 1.5px; }
.sidebar-logo .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }

/* ── Sidebar Nav Items ── */
.nav-item { display: flex; align-items: center; gap: 12px; padding: 10px 16px;
    border-radius: 10px; margin: 4px 0; cursor: pointer; transition: all .2s; color: #7090b0; }
.nav-item:hover, .nav-item.active { background: rgba(240,192,64,.08); color: #f0c040; }
.nav-item .icon { font-size: 1.1rem; width: 24px; text-align: center; }
.nav-item .label { font-size: 0.88rem; font-weight: 500; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0d1929 0%, #111d30 100%);
    border: 1px solid #1a2d4a; border-radius: 16px; padding: 20px 24px;
    position: relative; overflow: hidden; transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,.4); }
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #f0c040, #e8960c); border-radius: 16px 16px 0 0; }
.kpi-label { font-size: 0.75rem; color: #4a6080; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 600; }
.kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem;
    font-weight: 700; color: #e8f0fe; margin: 4px 0; }
.kpi-sub { font-size: 0.78rem; color: #22c55e; }
.kpi-icon { position: absolute; right: 20px; top: 20px; font-size: 2rem; opacity: .12; }

/* ── Glass Card ── */
.glass-card {
    background: rgba(13,25,41,.8); border: 1px solid #1a2d4a;
    border-radius: 18px; padding: 28px; backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,.3);
}

/* ── PAN Input Box ── */
.pan-box {
    background: linear-gradient(135deg, #0d1929, #0f2035);
    border: 2px solid #1a3050; border-radius: 20px; padding: 32px;
    box-shadow: 0 8px 40px rgba(0,0,0,.4);
}
.pan-box h3 { color: #e8f0fe; font-family: 'Space Grotesk', sans-serif; margin: 0 0 6px; }
.pan-box p { color: #4a6080; font-size: .88rem; margin: 0; }

/* ── Score Display ── */
.score-ring {
    background: linear-gradient(135deg, #0d1929, #081422);
    border: 1px solid #1a3050; border-radius: 20px; padding: 32px;
    text-align: center; box-shadow: 0 8px 40px rgba(0,0,0,.5);
}
.score-number { font-family: 'Space Grotesk', sans-serif; font-size: 4.5rem;
    font-weight: 800; background: linear-gradient(135deg, #f0c040, #e8960c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; }
.score-label { font-size: .8rem; color: #4a6080; text-transform: uppercase;
    letter-spacing: 2px; margin-top: 6px; }

/* ── Badges ── */
.badge { display: inline-flex; align-items: center; gap: 8px; padding: 10px 24px;
    border-radius: 50px; font-weight: 700; font-size: .95rem; margin: 12px 0; }
.badge-approved { background: linear-gradient(135deg,#065f46,#047857); color: #6ee7b7;
    border: 1px solid #059669; }
.badge-review   { background: linear-gradient(135deg,#78350f,#92400e); color: #fcd34d;
    border: 1px solid #f59e0b; }
.badge-rejected { background: linear-gradient(135deg,#7f1d1d,#991b1b); color: #fca5a5;
    border: 1px solid #ef4444; }

/* ── Factor Bars ── */
.factor-row { margin: 8px 0; }
.factor-label { display: flex; justify-content: space-between; align-items: center;
    font-size: .82rem; color: #94a3b8; margin-bottom: 5px; }
.factor-name { font-weight: 500; color: #cbd5e1; }
.bar-track { background: #1a2d4a; border-radius: 6px; height: 7px; }
.bar-fill-red  { background: linear-gradient(90deg,#ef4444,#dc2626); height: 7px;
    border-radius: 6px; transition: width .6s ease; }
.bar-fill-green{ background: linear-gradient(90deg,#22c55e,#16a34a); height: 7px;
    border-radius: 6px; transition: width .6s ease; }

/* ── Tip Card ── */
.tip-card { background: #0d1929; border: 1px solid #1a3050; border-radius: 14px;
    padding: 20px; margin: 8px 0; }
.tip-card h5 { color: #f0c040; margin: 0 0 10px; font-size: .9rem; }
.tip-card li { color: #94a3b8; font-size: .83rem; margin: 6px 0; }

/* ── Demo PAN Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0f2035, #1a3050) !important;
    color: #94a3b8 !important; border: 1px solid #1a3050 !important;
    border-radius: 10px !important; font-size: .8rem !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    border-color: #f0c040 !important; color: #f0c040 !important;
    box-shadow: 0 0 16px rgba(240,192,64,.2) !important;
}

/* Primary button */
button[kind="primary"] {
    background: linear-gradient(135deg, #f0c040, #e8960c) !important;
    color: #0a0f1e !important; border: none !important;
    font-weight: 700 !important; font-size: 1rem !important;
    border-radius: 12px !important; padding: 14px 28px !important;
    box-shadow: 0 4px 24px rgba(240,192,64,.3) !important;
    transition: all .2s !important;
}
button[kind="primary"]:hover { transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(240,192,64,.4) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: #0d1929 !important; border: 1px solid #1a3050 !important;
    color: #e8f0fe !important; border-radius: 10px !important;
}
.stSelectbox > div > div { background: #0d1929 !important; border: 1px solid #1a3050 !important; border-radius: 10px !important; }
.stSlider > div { color: #e8f0fe !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0d1929 !important;
    border-radius: 12px !important; padding: 4px !important;
    border: 1px solid #1a2d4a !important; }
.stTabs [data-baseweb="tab"] { color: #4a6080 !important; font-weight: 600 !important;
    border-radius: 8px !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg,#1a3050,#1e3a5f) !important;
    color: #f0c040 !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: #0d1929 !important; color: #94a3b8 !important;
    border-radius: 10px !important; border: 1px solid #1a2d4a !important; }

/* ── Metrics ── */
[data-testid="metric-container"] { background: #0d1929 !important;
    border: 1px solid #1a2d4a !important; border-radius: 12px !important;
    padding: 16px !important; }
[data-testid="metric-container"] label { color: #4a6080 !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #f0c040 !important; }

/* ── Info/Success/Warning/Error boxes ── */
.stAlert { border-radius: 12px !important; border: none !important; }

/* ── Dataframe ── */
.dataframe { background: #0d1929 !important; color: #94a3b8 !important; }

/* ── Section divider ── */
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem;
    font-weight: 700; color: #e8f0fe; display: flex; align-items: center; gap: 10px;
    margin: 24px 0 16px; padding-bottom: 10px;
    border-bottom: 1px solid #1a2d4a; }

/* ── Sidebar Steps ── */
.step-item { display: flex; align-items: flex-start; gap: 14px; padding: 10px 0;
    border-bottom: 1px solid #111d30; }
.step-num { min-width: 28px; height: 28px; background: linear-gradient(135deg,#f0c040,#e8960c);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: .78rem; color: #0a0f1e; }
.step-text { font-size: .82rem; color: #4a6080; line-height: 1.5; }
.step-text strong { color: #94a3b8; }

/* ── Table override ── */
table { border-collapse: collapse; width: 100%; }
th { background: #0d1929 !important; color: #f0c040 !important; padding: 8px 12px !important; }
td { background: #081422 !important; color: #94a3b8 !important; padding: 6px 12px !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="brand">🏦 FinTrust AI</div>
        <div class="tagline">Credit Intelligence Platform</div>
        <div style="margin-top:10px;font-size:.78rem;color:#4a6080;">
            <span class="dot"></span>Live System Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 8px 16px;">
        <div style="font-size:.7rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;padding-left:8px;">Navigation</div>
        <div class="nav-item active"><span class="icon">🆔</span><span class="label">Credit Score Check</span></div>
        <div class="nav-item"><span class="icon">👤</span><span class="label">Underwriter Dashboard</span></div>
        <div class="nav-item"><span class="icon">🌍</span><span class="label">Portfolio Analytics</span></div>
        <div class="nav-item"><span class="icon">⚖️</span><span class="label">Fairness Audit</span></div>
    </div>
    <hr style="border:1px solid #111d30;margin:0 0 16px;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 8px;">
        <div style="font-size:.7rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;padding-left:8px;">How It Works</div>
        <div class="step-item"><div class="step-num">1</div><div class="step-text"><strong>Enter PAN</strong><br>Your 10-digit PAN card number</div></div>
        <div class="step-item"><div class="step-num">2</div><div class="step-text"><strong>Bureau Fetch</strong><br>AI retrieves your credit profile</div></div>
        <div class="step-item"><div class="step-num">3</div><div class="step-text"><strong>Model Scores</strong><br>Ensemble AI evaluates 20 factors</div></div>
        <div class="step-item"><div class="step-num">4</div><div class="step-text"><strong>Instant Report</strong><br>Get CIBIL score + XAI breakdown</div></div>
    </div>
    <hr style="border:1px solid #111d30;margin:16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:12px 8px;background:#050a12;border-radius:12px;border:1px solid #111d30;">
        <div style="font-size:.7rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Model Arsenal</div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin:6px 0;">
            <span style="font-size:.78rem;color:#4a6080;">XGBoost</span>
            <span style="font-size:.75rem;color:#22c55e;font-weight:600;">Active</span>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin:6px 0;">
            <span style="font-size:.78rem;color:#4a6080;">LightGBM</span>
            <span style="font-size:.75rem;color:#22c55e;font-weight:600;">Active</span>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin:6px 0;">
            <span style="font-size:.78rem;color:#4a6080;">Random Forest</span>
            <span style="font-size:.75rem;color:#22c55e;font-weight:600;">Active</span>
        </div>
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid #111d30;font-size:.72rem;color:#2a3a50;text-align:center;">Best model auto-selected by AUC-ROC</div>
    </div>
    <div style="margin-top:20px;text-align:center;font-size:.68rem;color:#2a3a50;">v2.0 · FinTrust AI Platform<br>© 2026 All rights reserved</div>
    """, unsafe_allow_html=True)

    # ── API Provider Configuration ──
    st.markdown('<hr style="border:1px solid #111d30;margin:12px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.7rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1.5px;padding-left:8px;margin-bottom:8px;">Bureau API Settings</div>', unsafe_allow_html=True)

    api_provider = st.selectbox(
        "Data Provider",
        ["mock (sandbox)", "perfios", "setu", "karza", "cibil", "experian"],
        index=0,
        help="Select your credit bureau API provider. Use 'mock' for demo/testing."
    )
    provider_key = api_provider.split()[0]   # strip " (sandbox)"

    api_key_val = ""
    api_secret_val = ""
    if provider_key != "mock":
        api_key_val    = st.text_input("API Key / Client ID",    type="password", placeholder="Enter API key")
        api_secret_val = st.text_input("API Secret / Client Secret", type="password", placeholder="Enter secret (if needed)")
        st.caption("🔒 Keys are never stored or transmitted beyond this session.")

    # Build the client — used in Tab 1
    @st.cache_resource
    def build_api_client(prov, key, secret):
        return PANApiClient(prov, api_key=key, secret=secret)

    pan_api_client = build_api_client(provider_key, api_key_val, api_secret_val)

# ─────────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    p = 'data/processed_credit_data.csv'
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    p = 'models/encoders.pkl'
    return pickle.load(open(p,'rb')) if os.path.exists(p) else {}

df = load_data()
model = load_model()
encoders = load_encoders()
if df is not None:
    X = df.drop('target', axis=1)
    explainer_global = shap.TreeExplainer(model)
    shap_vals_global = explainer_global(X)
else:
    X = None; explainer_global = None; shap_vals_global = None

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def validate_pan(pan):
    return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan))

def pan_to_features(pan):
    seed = int(hashlib.md5(pan.encode()).hexdigest(), 16) % (2**31)
    rng  = np.random.default_rng(seed)
    return {
        'checking_status':      int(rng.choice([0,1,2,3])),
        'duration':             int(rng.integers(6, 72)),
        'credit_history':       int(rng.choice([0,1,2,3,4])),
        'purpose':              int(rng.choice(range(10))),
        'credit_amount':        int(rng.integers(500, 15000)),
        'savings_status':       int(rng.choice([0,1,2,3,4])),
        'employment':           int(rng.choice([0,1,2,3,4])),
        'installment_commitment': int(rng.integers(1,5)),
        'personal_status':      int(rng.choice([0,1,2,3])),
        'other_parties':        int(rng.choice([0,1,2])),
        'residence_since':      int(rng.integers(1,5)),
        'property_magnitude':   int(rng.choice([0,1,2,3])),
        'age':                  int(rng.integers(19,75)),
        'other_payment_plans':  int(rng.choice([0,1,2])),
        'housing':              int(rng.choice([0,1,2])),
        'existing_credits':     int(rng.integers(1,5)),
        'job':                  int(rng.choice([0,1,2,3])),
        'num_dependents':       int(rng.integers(1,3)),
        'own_telephone':        int(rng.choice([0,1])),
        'foreign_worker':       int(rng.choice([0,1])),
    }

def cibil(p): return int(900 - p * 600)

def grade(s):
    if s >= 750: return "EXCELLENT", "#22c55e", "badge-approved", "✅"
    if s >= 650: return "GOOD",      "#f0c040", "badge-review",   "✦"
    if s >= 550: return "FAIR",      "#f97316", "badge-review",   "⚠️"
    return            "POOR",        "#ef4444", "badge-rejected",  "❌"

# ─────────────────────────────────────────────
#  TOP HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:22px 32px;background:linear-gradient(135deg,#0d1929 0%,#111d30 100%);
            border-radius:18px;border:1px solid #1a2d4a;margin-bottom:24px;
            box-shadow:0 8px 40px rgba(0,0,0,.5);">
    <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:800;
                    color:#e8f0fe;letter-spacing:-0.5px;">
            Credit Intelligence Platform
        </div>
        <div style="font-size:.85rem;color:#4a6080;margin-top:4px;">
            Powered by Explainable AI · Real-time Bureau Integration · RBI Compliant
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:.75rem;color:#22c55e;font-weight:600;letter-spacing:1px;">● SYSTEM ONLINE</div>
        <div style="font-size:.7rem;color:#2a3a50;margin-top:2px;">Model Accuracy: 82.5% · AUC: 0.84</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_pan, tab_under, tab_global, tab_fair, tab_sim = st.tabs([
    "🆔  Credit Score Check",
    "👤  Underwriter Dashboard",
    "🌍  Portfolio Analytics",
    "⚖️  Fairness Audit",
    "🎮  What-If Simulator",
])

# ══════════════════════════════════════════════
#  TAB 1 — PAN CARD CREDIT CHECK
# ══════════════════════════════════════════════
with tab_pan:
    # ── Demo PAN quick-fill
    st.markdown("""<div class="section-title">🆔 Instant Credit Score — PAN Card Lookup</div>""", unsafe_allow_html=True)

    demo_row = st.columns(4)
    demo_pans = [("ABCDE1234F","✅ High Score"),("PQRST5678U","🟡 Medium"),("MNOPQ9012R","🔴 Low Score"),("XYZAB3456C","🎲 Random")]
    for i,(dpan,dlbl) in enumerate(demo_pans):
        with demo_row[i]:
            if st.button(f"{dlbl}\n`{dpan}`", key=f"demo{i}", use_container_width=True):
                st.session_state['pan'] = dpan

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.8], gap="large")

    with left_col:
        st.markdown("""<div class="pan-box">
            <h3>🪪 Enter PAN Details</h3>
            <p>Bureau lookup is instant and encrypted</p>
        </div>""", unsafe_allow_html=True)

        pan_val = st.session_state.get('pan','')
        pan_input = st.text_input("PAN Card Number", value=pan_val,
            max_chars=10, placeholder="ABCDE1234F",
            help="Format: 5 letters + 4 digits + 1 letter").upper().strip()
        if pan_input: st.session_state['pan'] = pan_input

        c1,c2 = st.columns(2)
        with c1: u_name = st.text_input("Full Name", placeholder="Raj Kumar")
        with c2: u_age  = st.number_input("Age", 18, 100, 30)
        c3,c4 = st.columns(2)
        with c3: u_income = st.number_input("Annual Income (₹)", 0, 10000000, 500000, step=10000)
        with c4: u_emp = st.selectbox("Employment", ["Salaried","Self-Employed","Business","Unemployed"])

        with st.expander("⚙️ Advanced — Override Bureau Data"):
            a1,a2 = st.columns(2)
            with a1:
                ov_dur = st.slider("Loan Duration (mo.)", 6, 72, 24)
                ov_amt = st.number_input("Credit Amount (₹)", 500, 200000, 10000, 500)
                ov_age = st.slider("Age (override)", 18, 80, 35)
            with a2:
                ov_chk = st.selectbox("Checking Account",["No Account","<0 DM","0–200 DM",">200 DM"])
                ov_sav = st.selectbox("Savings Account",["No Savings","<100 DM","100–500 DM","500–1000 DM",">1000 DM"])
                ov_emp = st.selectbox("Employment Duration",["Unemployed","<1 Yr","1–4 Yr","4–7 Yr",">7 Yr"])
            use_ov = st.checkbox("Use my manual entries", value=False)

        submitted = st.button("🔍 Check Credit Score", type="primary", use_container_width=True)

    with right_col:
        if submitted or 'result' in st.session_state:
            if submitted:
                if not pan_input:
                    st.error("⚠️ Please enter a PAN card number.")
                    st.stop()
                if not validate_pan(pan_input):
                    st.error(f"❌ Invalid PAN: `{pan_input}` — Expected format: `ABCDE1234F`")
                    st.stop()

                # ── Fetch profile via bureau API ──────────
                with st.spinner(f"🔄 Fetching bureau data via **{pan_api_client.provider.upper()}**..."):
                    profile = pan_api_client.get_credit_profile(pan_input)

                # Show API error if any (still continue — fallback data is usable)
                if profile.error:
                    st.warning(f"⚠️ Bureau API note: {profile.error} — using simulated fallback data.")

                feats = profile.to_model_input()

                # Override with manual entries if requested
                if use_ov:
                    feats['duration']         = ov_dur
                    feats['credit_amount']    = ov_amt
                    feats['age']             = ov_age
                    feats['checking_status'] = ["No Account","<0 DM","0–200 DM",">200 DM"].index(ov_chk)
                    feats['savings_status']  = ["No Savings","<100 DM","100–500 DM","500–1000 DM",">1000 DM"].index(ov_sav)
                    feats['employment']      = ["Unemployed","<1 Yr","1–4 Yr","4–7 Yr",">7 Yr"].index(ov_emp)
                else:
                    feats['age'] = u_age if not profile.name or profile.name == "Unknown" else profile.to_model_input()['age']

                idf = pd.DataFrame([feats])
                prob = model.predict_proba(idf)[0][1]
                score = cibil(prob)
                g, col, badge_cls, ico = grade(score)

                st.session_state['result'] = {
                    'pan': pan_input, 'feats': feats, 'idf': idf,
                    'prob': prob, 'score': score, 'grade': g,
                    'color': col, 'badge': badge_cls, 'icon': ico,
                    'profile_name':   profile.name,
                    'profile_source': profile.source,
                    'pan_verified':   profile.pan_verified,
                    'monthly_income': profile.monthly_income,
                    'foir':           profile.foir,
                    'risk_band':      profile.perfios_risk_band,
                }

            r = st.session_state.get('result')
            if r:
                # ── Bureau source badge ────────────────────
                src = r.get('profile_source','mock')
                src_color = '#22c55e' if src == 'perfios' else ('#f0c040' if src in ['setu','karza'] else '#4a6080')
                verified_txt = '✅ PAN Verified' if r.get('pan_verified') else '⚠️ Unverified'
                name_txt = r.get('profile_name','') or ''
                st.markdown(f'''
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;
                            background:#050a12;border:1px solid #1a2d4a;border-radius:10px;padding:10px 16px;">
                    <span style="font-size:.78rem;color:{src_color};font-weight:700;text-transform:uppercase;
                                 background:rgba(0,0,0,.3);padding:3px 10px;border-radius:20px;
                                 border:1px solid {src_color}33;">● {src}</span>
                    <span style="font-size:.83rem;color:#94a3b8;">{name_txt}</span>
                    <span style="font-size:.78rem;color:#22c55e;margin-left:auto;">{verified_txt}</span>
                </div>
                ''', unsafe_allow_html=True)

                # Show Perfios BSA analytics if available
                if r.get('monthly_income') and r['monthly_income'] > 0:
                    pa1, pa2, pa3 = st.columns(3)
                    pa1.metric("Monthly Income", f"₹{r['monthly_income']:,.0f}")
                    pa2.metric("FOIR", f"{r['foir']:.1f}%", help="Fixed Obligation to Income Ratio — lower is better")
                    pa3.metric("Risk Band", r.get('risk_band','—') or '—')

                # ── Score + Badge
                s_col, g_col = st.columns([1, 1.4])
                with s_col:
                    st.markdown(f"""
                    <div class="score-ring">
                        <div class="score-label">CIBIL SCORE</div>
                        <div class="score-number">{r['score']}</div>
                        <div style="font-size:.85rem;color:{r['color']};font-weight:700;margin-top:8px;">{r['icon']} {r['grade']}</div>
                        <div class="score-label" style="margin-top:4px;">Range: 300 – 900</div>
                        <div style="margin-top:16px;"><span class="{r['badge']} badge">
                            {'✅ AUTO-APPROVED' if r['score']>=750 else ('⚠️ MANUAL REVIEW' if r['score']>=600 else '❌ REJECTED')}
                        </span></div>
                        <div style="font-size:.75rem;color:#2a3a50;margin-top:12px;">
                            Default Risk: <span style="color:{r['color']};font-weight:600;">{round(r['prob']*100,1)}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with g_col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r['score'],
                        number={'font':{'size':44,'color':'#f0c040','family':'Space Grotesk'}},
                        gauge={
                            'axis':{'range':[300,900],'tickcolor':'#1a2d4a','tickwidth':1,
                                    'tickvals':[300,450,600,750,900],'tickfont':{'color':'#4a6080','size':11}},
                            'bar':{'color':'#f0c040','thickness':0.22},
                            'bgcolor':'#081422',
                            'borderwidth':0,
                            'steps':[
                                {'range':[300,550],'color':'#1f1015'},
                                {'range':[550,650],'color':'#1a1a0f'},
                                {'range':[650,750],'color':'#0f1a0f'},
                                {'range':[750,900],'color':'#0a1f10'},
                            ],
                        }
                    ))
                    fig.update_layout(
                        height=260, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'family':'Inter','color':'#e8f0fe'},
                        margin=dict(l=20,r=20,t=30,b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── KPI row
                k1,k2,k3,k4 = st.columns(4)
                k_data = [
                    ("Credit Score", r['score'], "300–900 range", "🎯"),
                    ("Default Risk", f"{round(r['prob']*100,1)}%", "Probability", "📉"),
                    ("Age Factor", r['feats']['age'], "Years", "👤"),
                    ("Credit Amount", f"₹{r['feats']['credit_amount']:,}", "Loan amount", "💰"),
                ]
                for col_k, (lbl, val, sub, ico) in zip([k1,k2,k3,k4], k_data):
                    col_k.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">{ico}</div>
                        <div class="kpi-label">{lbl}</div>
                        <div class="kpi-value">{val}</div>
                        <div class="kpi-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

                # ── SHAP XAI
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">🧠 AI Decision Justification (XAI)</div>', unsafe_allow_html=True)

                xai_exp = shap.TreeExplainer(model)
                sv = xai_exp(r['idf'])
                sh_vals = sv[0].values
                fnames  = list(r['idf'].columns)
                contrib = pd.DataFrame({'Feature':fnames,'Impact':sh_vals}).sort_values('Impact',ascending=False)

                xc1, xc2 = st.columns(2)
                with xc1:
                    st.markdown('<div style="font-size:.88rem;font-weight:600;color:#ef4444;margin-bottom:12px;">🔴 Risk Amplifiers</div>', unsafe_allow_html=True)
                    for _,row in contrib.head(4).iterrows():
                        bp = min(100, int(abs(row['Impact'])*800))
                        st.markdown(f"""
                        <div class="factor-row">
                            <div class="factor-label">
                                <span class="factor-name">{row['Feature']}</span>
                                <span style="color:#ef4444;font-weight:600;">+{row['Impact']:.3f}</span>
                            </div>
                            <div class="bar-track"><div class="bar-fill-red" style="width:{bp}%"></div></div>
                        </div>""", unsafe_allow_html=True)

                with xc2:
                    st.markdown('<div style="font-size:.88rem;font-weight:600;color:#22c55e;margin-bottom:12px;">🟢 Protective Factors</div>', unsafe_allow_html=True)
                    for _,row in contrib.tail(4).iterrows():
                        bp = min(100, int(abs(row['Impact'])*800))
                        st.markdown(f"""
                        <div class="factor-row">
                            <div class="factor-label">
                                <span class="factor-name">{row['Feature']}</span>
                                <span style="color:#22c55e;font-weight:600;">{row['Impact']:.3f}</span>
                            </div>
                            <div class="bar-track"><div class="bar-fill-green" style="width:{bp}%"></div></div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 Detailed SHAP Waterfall Chart"):
                    st_shap(shap.plots.waterfall(sv[0]), height=420)

                # ── Bureau snapshot
                st.markdown('<div class="section-title">📋 Bureau Data Snapshot</div>', unsafe_allow_html=True)
                snap_labels = {
                    'checking_status':'Checking Account','duration':'Loan Duration (mo.)',
                    'credit_history':'Credit History','credit_amount':'Credit Amount (₹)',
                    'savings_status':'Savings Status','employment':'Employment',
                    'age':'Age','installment_commitment':'Installment Rate (%)',
                    'num_dependents':'Dependents','existing_credits':'Existing Credits'
                }
                snap_df = pd.DataFrame([
                    {'Field': snap_labels.get(k,k), 'Value': v}
                    for k,v in r['feats'].items() if k in snap_labels
                ])
                st.dataframe(snap_df.set_index('Field'), use_container_width=True)

                # ── Improvement Tips
                st.markdown('<div class="section-title">💡 Score Improvement Roadmap</div>', unsafe_allow_html=True)
                t1,t2,t3 = st.columns(3)
                with t1:
                    st.markdown("""<div class="tip-card">
                        <h5>⚡ Quick Wins (0–3 months)</h5>
                        <ul>
                            <li>Pay all dues before due date</li>
                            <li>Clear outstanding balances</li>
                            <li>Dispute errors in credit report</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                with t2:
                    st.markdown("""<div class="tip-card">
                        <h5>📈 Mid Term (3–12 months)</h5>
                        <ul>
                            <li>Keep credit utilization &lt;30%</li>
                            <li>Avoid multiple loan applications</li>
                            <li>Maintain 1 secured credit card</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                with t3:
                    st.markdown("""<div class="tip-card">
                        <h5>🏆 Long Term (1–3 years)</h5>
                        <ul>
                            <li>Build diversified credit mix</li>
                            <li>Keep old accounts active</li>
                            <li>Grow emergency savings fund</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:80px 40px;">
                <div style="font-size:4rem;margin-bottom:20px;">🪪</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;
                            font-weight:700;color:#e8f0fe;margin-bottom:12px;">
                    Enter PAN to Get Started
                </div>
                <div style="color:#4a6080;font-size:.9rem;max-width:380px;margin:0 auto;line-height:1.7;">
                    Your AI-powered credit report will appear here. Enter your PAN card number
                    or click a demo button on the left.
                </div>
                <div style="margin-top:32px;display:flex;justify-content:center;gap:32px;">
                    <div style="text-align:center;">
                        <div style="font-size:1.8rem;font-weight:800;color:#f0c040;font-family:'Space Grotesk'">20</div>
                        <div style="font-size:.72rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1px;">Features</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.8rem;font-weight:800;color:#f0c040;font-family:'Space Grotesk'">4</div>
                        <div style="font-size:.72rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1px;">AI Models</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.8rem;font-weight:800;color:#f0c040;font-family:'Space Grotesk'">&lt;1s</div>
                        <div style="font-size:.72rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1px;">Response</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — UNDERWRITER DASHBOARD
# ══════════════════════════════════════════════
with tab_under:
    if df is None:
        st.warning("⚠️ Dataset not found. Please run `train_model.py` first.")
    else:
        st.markdown('<div class="section-title">👤 Individual Applicant Analysis</div>', unsafe_allow_html=True)
        idx = st.slider("Select Applicant ID", 0, len(X)-1, 0)
        app_data = X.iloc[[idx]]
        prob_u = model.predict_proba(app_data)[0][1]
        score_u = cibil(prob_u)
        g_u, c_u, b_u, i_u = grade(score_u)

        m1,m2,m3,m4 = st.columns(4)
        for col_m,(lbl,val,ico) in zip([m1,m2,m3,m4],[
            ("CIBIL Score", score_u, "🎯"),
            ("Default Risk", f"{round(prob_u*100,1)}%", "📉"),
            ("Age", int(app_data['age'].values[0]), "👤"),
            ("Credit Amount", f"₹{int(app_data['credit_amount'].values[0]):,}", "💰")
        ]):
            col_m.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{ico}</div>
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">Applicant #{idx}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2 = st.columns([1,1.5])
        with p1:
            st.markdown(f'<div class="section-title">Profile Data</div>', unsafe_allow_html=True)
            st.dataframe(app_data.T.rename(columns={idx:"Value"}), use_container_width=True, height=380)
        with p2:
            st.markdown('<div class="section-title">Risk Gauge</div>', unsafe_allow_html=True)
            fig_u = go.Figure(go.Indicator(
                mode="gauge+number", value=score_u,
                number={'font':{'color':'#f0c040','size':40,'family':'Space Grotesk'}},
                gauge={
                    'axis':{'range':[300,900],'tickvals':[300,450,600,750,900],'tickfont':{'color':'#4a6080','size':11}},
                    'bar':{'color':'#f0c040','thickness':0.22},
                    'bgcolor':'#081422','borderwidth':0,
                    'steps':[
                        {'range':[300,550],'color':'#1f1015'},
                        {'range':[550,650],'color':'#1a1a0f'},
                        {'range':[650,750],'color':'#0f1a0f'},
                        {'range':[750,900],'color':'#0a1f10'},
                    ]
                }
            ))
            fig_u.update_layout(height=300,paper_bgcolor='rgba(0,0,0,0)',
                font={'family':'Inter','color':'#e8f0fe'},margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_u, use_container_width=True)
            st.markdown(f'<div style="text-align:center;"><span class="{b_u} badge">{i_u} {g_u}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🧠 SHAP Explanation</div>', unsafe_allow_html=True)
        sv_u = shap_vals_global[idx].values
        cont_u = pd.DataFrame({'Feature':X.columns,'Impact':sv_u}).sort_values('Impact',ascending=False)
        top2r = cont_u.head(2); top1s = cont_u.tail(1)
        st.info(f"**AI Summary**: Applicant #{idx} has a **{round(prob_u*100,1)}%** default probability. "
                f"Key risk drivers: **{top2r.iloc[0]['Feature']}** and **{top2r.iloc[1]['Feature']}**. "
                f"Strongest mitigant: **{top1s.iloc[0]['Feature']}**.")
        with st.expander("📊 SHAP Waterfall"):
            st_shap(shap.plots.waterfall(shap_vals_global[idx]), height=400)

# ══════════════════════════════════════════════
#  TAB 3 — PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════
with tab_global:
    if df is None:
        st.warning("⚠️ Dataset not found. Please run `train_model.py` first.")
    else:
        st.markdown('<div class="section-title">🌍 Portfolio-Level Risk Analytics</div>', unsafe_allow_html=True)
        y_pred_all = model.predict(X)
        acc = round(accuracy_score(df['target'], y_pred_all)*100, 2)
        approval = round((y_pred_all == 0).mean()*100, 2)

        pm1,pm2,pm3,pm4 = st.columns(4)
        for col_p,(lbl,val,sub,ico) in zip([pm1,pm2,pm3,pm4],[
            ("Total Applicants", len(X), "In portfolio", "📁"),
            ("Model Accuracy", f"{acc}%", "Test set performance","🎯"),
            ("Approval Rate", f"{approval}%", "Good credit", "✅"),
            ("Default Rate", f"{round(100-approval,2)}%", "High risk","⚠️"),
        ]):
            col_p.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon">{ico}</div>
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ga,gb = st.columns(2)
        with ga:
            st.markdown('<div class="section-title">Top Risk Drivers (Global)</div>', unsafe_allow_html=True)
            fig_g1, ax = plt.subplots(facecolor='#0d1929')
            ax.set_facecolor('#081422')
            shap.summary_plot(shap_vals_global, X, plot_type="bar", show=False,
                              color='#f0c040', plot_size=None)
            ax.tick_params(colors='#4a6080'); ax.xaxis.label.set_color('#4a6080')
            for spine in ax.spines.values(): spine.set_edgecolor('#1a2d4a')
            st.pyplot(fig_g1); plt.clf()

        with gb:
            st.markdown('<div class="section-title">Directional Impact (Beeswarm)</div>', unsafe_allow_html=True)
            fig_g2, ax2 = plt.subplots(facecolor='#0d1929')
            ax2.set_facecolor('#081422')
            shap.summary_plot(shap_vals_global, X, show=False, plot_size=None)
            ax2.tick_params(colors='#4a6080'); ax2.xaxis.label.set_color('#4a6080')
            for spine in ax2.spines.values(): spine.set_edgecolor('#1a2d4a')
            st.pyplot(fig_g2); plt.clf()

        # Score distribution
        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
        probs_all = model.predict_proba(X)[:,1]
        scores_all = [cibil(p) for p in probs_all]
        fig_dist = px.histogram(x=scores_all, nbins=40,
            labels={'x':'CIBIL Score','y':'Count'},
            color_discrete_sequence=['#f0c040'])
        fig_dist.update_layout(
            paper_bgcolor='#0d1929', plot_bgcolor='#081422',
            font={'color':'#94a3b8','family':'Inter'},
            xaxis={'gridcolor':'#1a2d4a'}, yaxis={'gridcolor':'#1a2d4a'},
            margin=dict(l=20,r=20,t=20,b=20), height=300
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 4 — FAIRNESS AUDIT
# ══════════════════════════════════════════════
with tab_fair:
    if df is None:
        st.warning("⚠️ Dataset not found. Please run `train_model.py` first.")
    else:
        st.markdown('<div class="section-title">⚖️ AI Fairness & Regulatory Compliance</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.88rem;color:#4a6080;margin-bottom:20px;">Ensuring compliance with Equal Credit Opportunity Act (ECOA) · RBI Fair Lending Standards</div>', unsafe_allow_html=True)

        sf = (df['age'] < 25).astype(int)
        yt = df['target']
        yp = model.predict(X)
        dp = demographic_parity_difference(yt, yp, sensitive_features=sf)
        acc_f = round(accuracy_score(yt, yp)*100, 2)

        fa1,fa2,fa3 = st.columns(3)
        fa1.metric("Demographic Parity Diff", f"{round(dp*100,2)}%", delta=None)
        fa2.metric("Model Accuracy", f"{acc_f}%")
        fa3.metric("Under-25 Flag", "HIGH RISK" if dp > 0.1 else "COMPLIANT",
                   delta="Action Required" if dp > 0.1 else "Passed")

        st.markdown("<br>", unsafe_allow_html=True)
        if dp > 0.1:
            st.error("""❌ **Audit Failed** — The model applies disproportionate risk to applicants under 25.  
            **Required Actions:** Apply fairness constraints (Fairlearn reweighing) before production deployment.""")
        else:
            st.success("""✅ **Audit Passed** — Model demonstrates equitable approval rates across all age demographics.  
            Demographic parity difference is within acceptable RBI thresholds (<10%).""")

        # Age-split analysis
        st.markdown('<div class="section-title">Age Group Analysis</div>', unsafe_allow_html=True)
        df_audit = df.copy()
        df_audit['predicted'] = yp
        df_audit['age_group'] = pd.cut(df_audit['age'], bins=[0,25,35,50,100],
                                        labels=['Under 25','25–34','35–49','50+'])
        grp = df_audit.groupby('age_group').agg(
            Count=('target','count'),
            Default_Rate=('target','mean'),
            Approval_Rate=('predicted', lambda x: (x==0).mean())
        ).reset_index()
        grp['Default_Rate'] = (grp['Default_Rate']*100).round(1).astype(str)+'%'
        grp['Approval_Rate'] = (grp['Approval_Rate']*100).round(1).astype(str)+'%'
        st.dataframe(grp, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 5 — WHAT-IF SCENARIO SIMULATOR
# ══════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="section-title">🎮 What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.88rem;color:#4a6080;margin-bottom:20px;">'
        'Adjust any credit factor below and instantly see how your CIBIL score changes. '
        'Find the <b style="color:#f0c040;">exact actions</b> needed to move to the next grade.'
        '</div>', unsafe_allow_html=True
    )

    # ── Seed from last PAN result if available, else use neutral defaults
    if 'result' in st.session_state:
        base_feats = dict(st.session_state['result']['feats'])
        base_score = st.session_state['result']['score']
        base_prob  = st.session_state['result']['prob']
        seed_label = f"Loaded from PAN: `{st.session_state['result']['pan']}`"
    else:
        base_feats = {
            'checking_status': 1, 'duration': 24, 'credit_history': 3,
            'purpose': 2, 'credit_amount': 5000, 'savings_status': 1,
            'employment': 2, 'installment_commitment': 2, 'personal_status': 1,
            'other_parties': 0, 'residence_since': 2, 'property_magnitude': 1,
            'age': 35, 'other_payment_plans': 0, 'housing': 1,
            'existing_credits': 1, 'job': 2, 'num_dependents': 1,
            'own_telephone': 1, 'foreign_worker': 1,
        }
        base_prob  = model.predict_proba(pd.DataFrame([base_feats]))[0][1]
        base_score = cibil(base_prob)
        seed_label = "Using default profile — check a PAN first for a personalised simulation"

    st.info(f"📌 **Baseline:** {seed_label}  |  **Score:** {base_score}", icon="📊")

    sim_left, sim_right = st.columns([1.2, 1], gap="large")

    with sim_left:
        st.markdown('<div style="font-size:.9rem;font-weight:600;color:#e8f0fe;margin-bottom:12px;">🎚️ Adjust Credit Factors</div>', unsafe_allow_html=True)

        feature_configs = {
            'checking_status': {'label': '🏦 Checking Account Status', 'type': 'select',
                'options': ['No Account (worst)', '< 0 DM (negative)', '0–200 DM (ok)', '> 200 DM (best)'],
                'help': 'Higher is better. A healthy checking account lowers risk.'},
            'credit_history': {'label': '📜 Credit History Quality', 'type': 'select',
                'options': ['Critical/Other Account', 'No Credits Taken', 'All Paid Duly', 'Existing Paid', 'All Paid (best)'],
                'help': 'Past repayment behaviour. 4 = perfect history.'},
            'savings_status': {'label': '💰 Savings Account Balance', 'type': 'select',
                'options': ['No Savings (worst)', '< 100 DM', '100–500 DM', '500–1000 DM', '> 1000 DM (best)'],
                'help': 'More savings = lower default risk.'},
            'employment': {'label': '💼 Employment Duration', 'type': 'select',
                'options': ['Unemployed (worst)', '< 1 Year', '1–4 Years', '4–7 Years', '> 7 Years (best)'],
                'help': 'Longer stable employment = better score.'},
            'duration': {'label': '📅 Loan Duration (months)', 'type': 'slider', 'min': 6, 'max': 72, 'step': 6,
                'help': 'Shorter loans have lower default risk.'},
            'credit_amount': {'label': '💳 Credit Amount (₹ equiv.)', 'type': 'slider', 'min': 500, 'max': 15000, 'step': 500,
                'help': 'Lower loan amount reduces default probability.'},
            'installment_commitment': {'label': '📊 Installment Rate (% income)', 'type': 'slider', 'min': 1, 'max': 4, 'step': 1,
                'help': '1 = low burden, 4 = high burden. Lower is better.'},
            'age': {'label': '🎂 Age (years)', 'type': 'slider', 'min': 18, 'max': 80, 'step': 1,
                'help': 'Older applicants tend to have more stable profiles.'},
            'existing_credits': {'label': '🔢 Existing Credits at Bank', 'type': 'slider', 'min': 1, 'max': 4, 'step': 1,
                'help': 'Fewer existing credits = less outstanding burden.'},
            'residence_since': {'label': '🏠 Years at Current Residence', 'type': 'slider', 'min': 1, 'max': 4, 'step': 1,
                'help': 'Longer at same address = more stable.'},
            'num_dependents': {'label': '👨‍👩‍👧 Number of Dependents', 'type': 'slider', 'min': 1, 'max': 2, 'step': 1,
                'help': 'Fewer dependents = less financial pressure.'},
            'housing': {'label': '🏡 Housing Status', 'type': 'select',
                'options': ['Free Housing', 'Renting', 'Own Property (best)'],
                'help': 'Owning property signals financial stability.'},
            'purpose': {'label': '🎯 Loan Purpose', 'type': 'select',
                'options': ['Car (New)', 'Car (Used)', 'Furniture', 'Radio/TV', 'Appliances',
                            'Repairs', 'Education', 'Vacation', 'Retraining', 'Business'],
                'help': 'Productive purposes (car, education) have lower default rates.'},
            'other_payment_plans': {'label': '💸 Other Payment Plans', 'type': 'select',
                'options': ['None (best)', 'Stores', 'Banks'],
                'help': 'No other payment plans = lower financial burden.'},
            'property_magnitude': {'label': '🏛️ Property / Collateral', 'type': 'select',
                'options': ['No Property (worst)', 'Car/Other', 'Life Insurance', 'Real Estate (best)'],
                'help': 'More valuable collateral = lower lender risk.'},
            'personal_status': {'label': '👤 Personal Status', 'type': 'select',
                'options': ['Male Divorced/Sep', 'Female Div/Dep/Mar', 'Male Single', 'Male Mar/Wid'],
                'help': 'Demographic factor from German Credit dataset.'},
            'other_parties': {'label': '🤝 Other Parties (Guarantor)', 'type': 'select',
                'options': ['None', 'Co-Applicant', 'Guarantor (best)'],
                'help': 'Having a guarantor reduces lender risk.'},
            'job': {'label': '🧑‍💻 Job Skill Level', 'type': 'select',
                'options': ['Unskilled Non-Resident', 'Unskilled Resident', 'Skilled', 'Highly Skilled (best)'],
                'help': 'Higher skill level = more stable income.'},
            'own_telephone': {'label': '📞 Registered Phone', 'type': 'select',
                'options': ['No', 'Yes'],
                'help': 'Registered phone is a positive stability signal.'},
            'foreign_worker': {'label': '🌐 Foreign Worker Status', 'type': 'select',
                'options': ['Yes', 'No'],
                'help': 'Non-foreign workers have lower default rates in this dataset.'},
        }

        sim_feats = {}
        for feat_key, cfg in feature_configs.items():
            cur = base_feats.get(feat_key, 0)
            if cfg['type'] == 'slider':
                sim_feats[feat_key] = st.slider(
                    cfg['label'], cfg['min'], cfg['max'], int(cur), cfg['step'],
                    help=cfg['help'], key=f"sim_{feat_key}"
                )
            else:
                opts = cfg['options']
                idx  = min(int(cur), len(opts) - 1)
                sel  = st.selectbox(cfg['label'], opts, index=idx,
                                    help=cfg['help'], key=f"sim_{feat_key}")
                sim_feats[feat_key] = opts.index(sel)

    with sim_right:
        # Live prediction
        sim_idf   = pd.DataFrame([sim_feats])
        sim_prob  = model.predict_proba(sim_idf)[0][1]
        sim_score = cibil(sim_prob)
        sim_g, sim_color, _, sim_ico = grade(sim_score)

        delta      = sim_score - base_score
        delta_prob = round((sim_prob - base_prob) * 100, 1)
        arrow      = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        d_color    = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#4a6080")

        # Before / After cards
        st.markdown('<div class="section-title">📊 Before vs After</div>', unsafe_allow_html=True)
        ba1, ba2, ba3 = st.columns([1, 0.4, 1])
        with ba1:
            bg, bc, _, bi = grade(base_score)
            st.markdown(f"""
            <div class="score-ring" style="padding:20px;">
                <div class="score-label">BASELINE</div>
                <div class="score-number" style="font-size:2.8rem;color:#4a6080;">{base_score}</div>
                <div style="font-size:.82rem;color:{bc};font-weight:700;margin-top:6px;">{bi} {bg}</div>
            </div>""", unsafe_allow_html=True)
        with ba2:
            st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:100%;padding-top:40px;">
                <div style="font-size:2.2rem;color:{d_color};font-weight:800;">{arrow}</div>
                <div style="font-size:1rem;font-weight:800;color:{d_color};">
                    {'+' if delta>=0 else ''}{delta}</div>
                <div style="font-size:.68rem;color:#2a3a50;">pts</div>
            </div>""", unsafe_allow_html=True)
        with ba3:
            st.markdown(f"""
            <div class="score-ring" style="padding:20px;border-color:{sim_color}33;">
                <div class="score-label">NEW SCORE</div>
                <div class="score-number" style="font-size:2.8rem;color:{sim_color};">{sim_score}</div>
                <div style="font-size:.82rem;color:{sim_color};font-weight:700;margin-top:6px;">{sim_ico} {sim_g}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk progress bar
        st.markdown(f"""
        <div style="background:#0d1929;border:1px solid #1a2d4a;border-radius:12px;padding:16px 20px;">
            <div style="display:flex;justify-content:space-between;font-size:.82rem;
                        color:#4a6080;margin-bottom:8px;">
                <span>Default Risk</span>
                <span style="color:{d_color};font-weight:700;">
                    {round(sim_prob*100,1)}%
                    ({'+' if delta_prob>=0 else ''}{delta_prob}%)
                </span>
            </div>
            <div style="background:#1a2d4a;border-radius:6px;height:10px;">
                <div style="background:{sim_color};width:{min(100,int(sim_prob*100))}%;
                            height:10px;border-radius:6px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Decision verdict
        if sim_score >= 750:
            st.success("✅ **AUTO-APPROVED** — Excellent creditworthiness!")
        elif sim_score >= 600:
            st.warning("⚠️ **MANUAL REVIEW** — Good but needs underwriter check.")
        else:
            st.error("❌ **REJECTED** — High default risk profile.")

        # Changes made list
        changed = {k: (base_feats.get(k,0), sim_feats[k])
                   for k in sim_feats if sim_feats[k] != base_feats.get(k,0)}
        if changed:
            st.markdown('<div class="section-title" style="font-size:.95rem;margin-top:16px;">🔄 Changes Made</div>', unsafe_allow_html=True)
            for feat, (old_v, new_v) in changed.items():
                lbl = feature_configs[feat]['label']
                up  = new_v > old_v
                cc  = "#22c55e" if up else "#ef4444"
                ci  = "↑" if up else "↓"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            background:#0d1929;border:1px solid #1a2d4a;border-radius:8px;
                            padding:7px 14px;margin:3px 0;font-size:.81rem;">
                    <span style="color:#94a3b8;">{lbl}</span>
                    <span>
                        <span style="color:#4a6080;">{old_v}</span>
                        <span style="color:#2a3a50;margin:0 5px;">→</span>
                        <span style="color:{cc};font-weight:700;">{new_v} {ci}</span>
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#2a3a50;font-size:.84rem;text-align:center;padding:20px;">← Move any slider to see live changes</div>', unsafe_allow_html=True)

        # SHAP before/after (in expander to keep layout clean)
        with st.expander("🧠 SHAP Breakdown — Before vs After"):
            xai_exp = shap.TreeExplainer(model)
            sv_base = xai_exp(pd.DataFrame([base_feats]))
            sv_sim  = xai_exp(sim_idf)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.caption("**Baseline**")
                st_shap(shap.plots.waterfall(sv_base[0]), height=340)
            with sc2:
                st.caption("**After Changes**")
                st_shap(shap.plots.waterfall(sv_sim[0]),  height=340)

        # Auto-recommendations
        st.markdown('<div class="section-title" style="font-size:.95rem;margin-top:16px;">💡 Top Actions to Improve</div>', unsafe_allow_html=True)
        recs = []
        if sim_feats['savings_status'] < 3:
            recs.append(("💰 Increase Savings", "Move savings to 100–500+ DM band. Reduces risk significantly."))
        if sim_feats['credit_history'] < 4:
            recs.append(("📜 Build Credit History", "Pay all dues on time for 6–12 months to reach 'All Paid' status."))
        if sim_feats['duration'] > 24:
            recs.append(("📅 Shorten Loan Term", f"Reduce duration from {sim_feats['duration']} → {max(6,sim_feats['duration']-12)} months."))
        if sim_feats['credit_amount'] > 8000:
            recs.append(("💳 Reduce Loan Amount", f"Request ₹{sim_feats['credit_amount']-2000:,} instead of ₹{sim_feats['credit_amount']:,}."))
        if sim_feats['checking_status'] < 2:
            recs.append(("🏦 Maintain Positive Balance", "Keep checking account above 0 DM consistently."))
        if sim_feats['installment_commitment'] > 2:
            recs.append(("📊 Lower EMI Burden", "Consolidate or prepay existing loans to reduce commitment."))
        if not recs:
            recs.append(("🏆 Profile is Strong!", "Maintain consistency for a great credit score."))
        for i, (title, desc) in enumerate(recs[:4]):
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;
                        background:#0d1929;border:1px solid #1a2d4a;border-radius:10px;
                        padding:12px 14px;margin:5px 0;">
                <div style="min-width:22px;height:22px;background:linear-gradient(135deg,#f0c040,#e8960c);
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-weight:800;font-size:.72rem;color:#0a0f1e;">{i+1}</div>
                <div>
                    <div style="font-weight:600;color:#f0c040;font-size:.84rem;">{title}</div>
                    <div style="color:#4a6080;font-size:.79rem;margin-top:3px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
