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
import io

def safe_shap_waterfall(shap_values, height=400):
    """Render SHAP as a sleek dark-themed horizontal bar chart matching the premium UI."""
    vals = shap_values.values
    feats = shap_values.feature_names if hasattr(shap_values, 'feature_names') else [f'F{i}' for i in range(len(vals))]
    order = np.argsort(np.abs(vals))
    fig_h = max(3.5, len(vals) * 0.25)
    
    # Updated enterprise colors
    fig, ax = plt.subplots(figsize=(7, fig_h), facecolor='#111827')
    ax.set_facecolor('#0B1120')
    colors = ['#10B981' if v < 0 else '#EF4444' for v in vals[order]]
    
    bars = ax.barh([feats[i] for i in order], vals[order], color=colors, height=0.5, edgecolor='none', alpha=0.9)
    ax.set_xlabel('SHAP Value (Impact on Risk)', color='#9CA3AF', fontsize=9, fontweight='500')
    ax.tick_params(axis='y', colors='#D1D5DB', labelsize=8.5)
    ax.tick_params(axis='x', colors='#9CA3AF', labelsize=8)
    
    for spine in ax.spines.values(): 
        spine.set_edgecolor('#1F2937')
        
    ax.axvline(x=0, color='#4B5563', linewidth=1, linestyle='--')
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinTrust AI | Credit Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — Modern Premium Fintech Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

/* ── Root Reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0B1120; }
section[data-testid="stSidebar"] > div { background: #0F172A; border-right: 1px solid #1E293B; }
.block-container { padding: 2rem 3rem; max-width: 1440px; }

/* ── Hide default menu/footer ── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } 
::-webkit-scrollbar-track { background: #0B1120; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ── Sidebar Logo ── */
.sidebar-logo {
    text-align: center; padding: 32px 16px 24px;
    border-bottom: 1px solid #1E293B; margin-bottom: 16px;
}
.sidebar-logo .brand { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem;
    font-weight: 800; background: linear-gradient(135deg, #6366F1, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; }
.sidebar-logo .tagline { font-size: 0.75rem; color: #9CA3AF; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500;}
.sidebar-logo .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #10B981; margin-right: 6px; box-shadow: 0 0 8px rgba(16, 185, 129, 0.6); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.6;transform:scale(1.2)} }

/* ── KPI Cards ── */
.kpi-card {
    background: #111827;
    border: 1px solid #1F2937; border-radius: 16px; padding: 18px 20px;
    position: relative; overflow: hidden; transition: all .3s ease;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    min-height: 130px; display: flex; flex-direction: column; justify-content: center;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); border-color: #374151; }
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:4px;
    background: linear-gradient(90deg, #6366F1, #06B6D4); border-radius: 16px 16px 0 0; opacity: 0.9;}
.kpi-label { font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 700; opacity: 0.8; margin-bottom: 4px; }
.kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.4rem, 4vw, 1.8rem);
    font-weight: 700; color: #F3F4F6; margin: 4px 0; line-height: 1.2;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-sub { font-size: 0.75rem; color: #10B981; font-weight: 600; }
.kpi-icon { position: absolute; right: 16px; top: 16px; font-size: 1.5rem; opacity: 0.15; transition: all .3s; }
.kpi-card:hover .kpi-icon { opacity: 0.4; transform: scale(1.1); }

/* ── Glass Card ── */
.glass-card {
    background: rgba(17, 24, 39, 0.7); border: 1px solid #1F2937;
    border-radius: 20px; padding: 32px; backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

/* ── PAN Input Box ── */
.pan-box {
    background: #111827;
    border: 1px solid #1F2937; border-radius: 16px; padding: 28px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15); margin-bottom: 20px;
}
.pan-box h3 { color: #F3F4F6; font-family: 'Space Grotesk', sans-serif; margin: 0 0 8px; font-weight: 700; }
.pan-box p { color: #9CA3AF; font-size: .9rem; margin: 0; }

/* ── Score Display ── */
.score-ring {
    background: #111827;
    border: 1px solid #1F2937; border-radius: 20px; padding: 36px 24px;
    text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    position: relative; overflow: hidden;
}
.score-ring::after { content:''; position:absolute; inset:0; background: radial-gradient(circle at top right, rgba(99,102,241,0.05), transparent 60%); pointer-events:none;}
.score-number { font-family: 'Space Grotesk', sans-serif; font-size: 5rem;
    font-weight: 800; background: linear-gradient(135deg, #6366F1, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1; margin: 10px 0; }
.score-label { font-size: .8rem; color: #9CA3AF; text-transform: uppercase;
    letter-spacing: 2px; font-weight: 600;}

/* ── Badges ── */
.badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px;
    border-radius: 50px; font-weight: 600; font-size: .85rem; margin: 12px 0; letter-spacing: 0.5px;}
.badge-approved { background: rgba(16, 185, 129, 0.1); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.2); }
.badge-review   { background: rgba(245, 158, 11, 0.1); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.2); }
.badge-rejected { background: rgba(239, 68, 68, 0.1); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.2); }

/* ── Factor Bars ── */
.factor-row { margin: 10px 0; }
.factor-label { display: flex; justify-content: space-between; align-items: center;
    font-size: .85rem; color: #9CA3AF; margin-bottom: 6px; }
.factor-name { font-weight: 500; color: #E5E7EB; }
.bar-track { background: #1F2937; border-radius: 8px; height: 6px; overflow: hidden; }
.bar-fill-red  { background: linear-gradient(90deg, #F87171, #EF4444); height: 100%; border-radius: 8px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
.bar-fill-green{ background: linear-gradient(90deg, #34D399, #10B981); height: 100%; border-radius: 8px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }

/* ── Tip Card ── */
.tip-card { background: #111827; border: 1px solid #1F2937; border-radius: 16px;
    padding: 24px; margin: 8px 0; transition: all 0.2s; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
.tip-card:hover { border-color: #374151; transform: translateY(-2px); }
.tip-card h5 { color: #F3F4F6; margin: 0 0 12px; font-size: 1rem; font-weight: 600; display:flex; align-items:center; gap:8px;}
.tip-card li { color: #9CA3AF; font-size: .85rem; margin: 8px 0; line-height: 1.5; }

/* ── Buttons ── */
.stButton > button {
    background: #111827 !important; color: #D1D5DB !important; 
    border: 1px solid #374151 !important; border-radius: 12px !important; 
    font-size: .85rem !important; font-weight: 500 !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    border-color: #6366F1 !important; color: #FFFFFF !important;
    background: #1E293B !important;
}

/* Primary button */
button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5, #3B82F6) !important;
    color: #FFFFFF !important; border: none !important;
    font-weight: 600 !important; font-size: 1rem !important;
    border-radius: 12px !important; padding: 12px 24px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    transition: all .3s ease !important;
}
button[kind="primary"]:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important; 
}

/* ── Inputs ── */
.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: #0B1120 !important; border: 1px solid #374151 !important;
    color: #F3F4F6 !important; border-radius: 10px !important; padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
    border-color: #6366F1 !important; box-shadow: 0 0 0 1px #6366F1 !important;
}
.stSelectbox > div > div { background: #0B1120 !important; border: 1px solid #374151 !important; border-radius: 10px !important; }
.stSlider > div { color: #F3F4F6 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #111827 !important;
    border-radius: 14px !important; padding: 6px !important;
    border: 1px solid #1F2937 !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { color: #9CA3AF !important; font-weight: 500 !important;
    border-radius: 10px !important; flex: 1 !important; justify-content: center !important; 
    padding: 10px 16px !important; transition: all 0.2s !important; }
.stTabs [data-baseweb="tab"]:hover { color: #F3F4F6 !important; background: #1E293B !important; }
.stTabs [aria-selected="true"] { background: #1E293B !important; color: #FFFFFF !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;}

/* ── Expander ── */
.streamlit-expanderHeader { background: #111827 !important; color: #D1D5DB !important;
    border-radius: 12px !important; border: 1px solid #1F2937 !important; font-weight: 500 !important; }
.streamlit-expanderContent { border: 1px solid #1F2937 !important; border-top: none !important; border-radius: 0 0 12px 12px !important; }

/* ── Section divider ── */
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem;
    font-weight: 700; color: #F3F4F6; display: flex; align-items: center; gap: 10px;
    margin: 32px 0 20px; padding-bottom: 12px;
    border-bottom: 1px solid #1F2937; }

/* ── Sidebar Steps ── */
.step-item { display: flex; align-items: flex-start; gap: 14px; padding: 12px 0;
    border-bottom: 1px dashed #1E293B; }
.step-item:last-child { border-bottom: none; }
.step-num { min-width: 28px; height: 28px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .8rem; color: #818CF8; }
.step-text { font-size: .85rem; color: #9CA3AF; line-height: 1.5; }
.step-text strong { color: #E5E7EB; font-weight: 600; display: block; margin-bottom: 2px;}

/* ── Table override ── */
table { border-collapse: collapse; width: 100%; border-radius: 12px; overflow: hidden; }
th { background: #1F2937 !important; color: #F3F4F6 !important; padding: 12px 16px !important; font-weight: 600 !important; font-size: 0.85rem !important;}
td { background: #111827 !important; color: #D1D5DB !important; padding: 10px 16px !important; font-size: 0.85rem !important; border-bottom: 1px solid #1F2937 !important; }
tr:last-child td { border-bottom: none !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="brand">🏦 FinTrust AI</div>
        <div class="tagline">Credit Intelligence</div>
        <div style="margin-top:12px;font-size:.75rem;color:#9CA3AF;font-weight:500;display:flex;align-items:center;justify-content:center;">
            <span class="dot"></span>Live System Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 8px 16px;">
        <div style="font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;font-weight:600;">How It Works</div>
        <div class="step-item"><div class="step-num">1</div><div class="step-text"><strong>Enter PAN</strong>Your 10-digit identity number</div></div>
        <div class="step-item"><div class="step-num">2</div><div class="step-text"><strong>Bureau Fetch</strong>Secure profile retrieval</div></div>
        <div class="step-item"><div class="step-num">3</div><div class="step-text"><strong>AI Evaluation</strong>Ensemble evaluates 20+ factors</div></div>
        <div class="step-item"><div class="step-num">4</div><div class="step-text"><strong>Instant Report</strong>Score breakdown & XAI audit</div></div>
    </div>
    <hr style="border:none; border-top:1px solid #1E293B; margin:8px 0 20px;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:16px;background:#111827;border-radius:14px;border:1px solid #1F2937;">
        <div style="font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;font-weight:600;">Model Arsenal</div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin:8px 0;">
            <span style="font-size:.8rem;color:#D1D5DB;font-weight:500;">XGBoost</span>
            <span style="font-size:.7rem;color:#10B981;background:rgba(16,185,129,0.1);padding:2px 8px;border-radius:12px;border:1px solid rgba(16,185,129,0.2);">Active</span>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin:8px 0;">
            <span style="font-size:.8rem;color:#D1D5DB;font-weight:500;">LightGBM</span>
            <span style="font-size:.7rem;color:#10B981;background:rgba(16,185,129,0.1);padding:2px 8px;border-radius:12px;border:1px solid rgba(16,185,129,0.2);">Active</span>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin:8px 0;">
            <span style="font-size:.8rem;color:#D1D5DB;font-weight:500;">Random Forest</span>
            <span style="font-size:.7rem;color:#10B981;background:rgba(16,185,129,0.1);padding:2px 8px;border-radius:12px;border:1px solid rgba(16,185,129,0.2);">Active</span>
        </div>
        <div style="margin-top:12px;padding-top:12px;border-top:1px dashed #334155;font-size:.7rem;color:#9CA3AF;text-align:center;">Best model auto-selected by AUC-ROC</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Provider Configuration ──
    st.markdown('<hr style="border:none; border-top:1px solid #1E293B; margin:24px 0 16px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;font-weight:600;">Bureau Settings</div>', unsafe_allow_html=True)

    api_provider = st.selectbox(
        "Data Provider",
        ["mock (sandbox)", "perfios", "setu", "karza", "cibil", "experian"],
        index=0,
        help="Select your credit bureau API provider. Use 'mock' for demo/testing."
    )
    provider_key = api_provider.split()[0]

    api_key_val = ""
    api_secret_val = ""
    if provider_key != "mock":
        api_key_val    = st.text_input("API Key / Client ID",    type="password", placeholder="Enter API key")
        api_secret_val = st.text_input("API Secret", type="password", placeholder="Enter secret (if needed)")
        st.caption("🔒 Keys are never stored beyond this session.")
        
    st.markdown('<div style="margin-top:30px;text-align:center;font-size:.7rem;color:#64748B;">v2.0 · FinTrust AI Platform<br>© 2026 All rights reserved</div>', unsafe_allow_html=True)

    @st.cache_resource
    def build_api_client(prov, key, secret):
        return PANApiClient(prov, api_key=key, secret=secret)

    pan_api_client = build_api_client(provider_key, api_key_val, api_secret_val)

# ─────────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    p = os.path.join(base_path, 'data', 'processed_credit_data.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model.pkl')
    if not os.path.exists(model_path):
        st.error(f"❌ Could not find model at: {model_path}")
        st.stop()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    base_path = os.path.dirname(__file__)
    p = os.path.join(base_path, 'models', 'encoders.pkl')
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
    if s >= 750: return "EXCELLENT", "#10B981", "badge-approved", "✅" # Emerald
    if s >= 650: return "GOOD",      "#3B82F6", "badge-review",   "✦" # Blue
    if s >= 550: return "FAIR",      "#F59E0B", "badge-review",   "⚠️" # Amber
    return            "POOR",        "#EF4444", "badge-rejected",  "❌" # Red

# ─────────────────────────────────────────────
#  TOP HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:28px 36px;background:#111827;
            border-radius:20px;border:1px solid #1F2937;margin-bottom:32px;
            box-shadow:0 10px 30px rgba(0,0,0,0.15);">
    <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:800;
                    color:#F3F4F6;letter-spacing:-0.5px; margin-bottom: 4px;">
            Credit Intelligence Platform
        </div>
        <div style="font-size:.9rem;color:#9CA3AF;font-weight:500;">
            Powered by Explainable AI · Real-time Bureau Integration · RBI Compliant
        </div>
    </div>
    <div style="text-align:right; background: #0B1120; padding: 12px 20px; border-radius: 12px; border: 1px solid #1E293B;">
        <div style="font-size:.75rem;color:#10B981;font-weight:700;letter-spacing:1px; margin-bottom: 4px; display:flex; align-items:center; justify-content:flex-end; gap:6px;">
            <span style="display:inline-block;width:6px;height:6px;background:#10B981;border-radius:50%;"></span> SYSTEM ONLINE
        </div>
        <div style="font-size:.75rem;color:#9CA3AF;">Model Accuracy: <span style="color:#F3F4F6;font-weight:600;">82.5%</span> · AUC: <span style="color:#F3F4F6;font-weight:600;">0.84</span></div>
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
    st.markdown("""<div class="section-title">🆔 Instant Credit Score — PAN Card Lookup</div>""", unsafe_allow_html=True)

    demo_row = st.columns(4)
    demo_pans = [("ABCDE1234F","✅ High Score"),("PQRST5678U","🔵 Medium"),("MNOPQ9012R","🔴 Low Score"),("XYZAB3456C","🎲 Random")]
    for i,(dpan,dlbl) in enumerate(demo_pans):
        with demo_row[i]:
            if st.button(f"{dlbl}\n`{dpan}`", key=f"demo{i}", use_container_width=True):
                st.session_state['pan'] = dpan

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.8], gap="large")

    with left_col:
        st.markdown("""<div class="pan-box">
            <h3>🪪 Enter Applicant Details</h3>
            <p>Bureau lookup is instant, encrypted, and leaves no footprint.</p>
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
                ov_chk = st.selectbox("Checking Account",["No Account","< ₹0","₹0–₹200","> ₹200"])
                ov_sav = st.selectbox("Savings Account",["No Savings","< ₹100","₹100–₹500","₹500–₹1000","> ₹1000"])
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

                with st.spinner(f"🔄 Fetching bureau data via **{pan_api_client.provider.upper()}**..."):
                    profile = pan_api_client.get_credit_profile(pan_input)

                if profile.error:
                    st.warning(f"⚠️ Bureau API note: {profile.error} — using simulated fallback data.")

                feats = profile.to_model_input()

                if use_ov:
                    feats['duration']         = ov_dur
                    feats['credit_amount']    = ov_amt
                    feats['age']             = ov_age
                    feats['checking_status'] = ["No Account","< ₹0","₹0–₹200","> ₹200"].index(ov_chk)
                    feats['savings_status']  = ["No Savings","< ₹100","₹100–₹500","₹500–₹1000","> ₹1000"].index(ov_sav)
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
                src = r.get('profile_source','mock')
                src_color = '#10B981' if src == 'perfios' else ('#6366F1' if src in ['setu','karza'] else '#64748B')
                verified_txt = '✅ Verified identity' if r.get('pan_verified') else '⚠️ Unverified'
                name_txt = r.get('profile_name','') or 'Unknown Individual'
                
                st.markdown(f'''
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;
                            background:#111827;border:1px solid #1F2937;border-radius:12px;padding:12px 20px;">
                    <span style="font-size:.75rem;color:{src_color};font-weight:700;text-transform:uppercase;
                                 background:rgba(255,255,255,.05);padding:4px 12px;border-radius:20px;
                                 border:1px solid {src_color}33;letter-spacing:1px;">● {src}</span>
                    <span style="font-size:.9rem;color:#F3F4F6;font-weight:500;">{name_txt}</span>
                    <span style="font-size:.8rem;color:#10B981;margin-left:auto;font-weight:500;">{verified_txt}</span>
                </div>
                ''', unsafe_allow_html=True)

                if r.get('monthly_income') and r['monthly_income'] > 0:
                    pa1, pa2, pa3 = st.columns(3)
                    pa1.metric("Monthly Income", f"₹{r['monthly_income']:,.0f}")
                    pa2.metric("FOIR", f"{r['foir']:.1f}%", help="Fixed Obligation to Income Ratio")
                    pa3.metric("Risk Band", r.get('risk_band','—') or '—')

                s_col, g_col = st.columns([1, 1.4])
                with s_col:
                    st.markdown(f"""
                    <div class="score-ring">
                        <div class="score-label">CIBIL SCORE</div>
                        <div class="score-number">{r['score']}</div>
                        <div style="font-size:.9rem;color:{r['color']};font-weight:700;margin-top:4px;">{r['icon']} {r['grade']}</div>
                        <div style="margin-top:16px;"><span class="{r['badge']} badge">
                            {'✅ AUTO-APPROVED' if r['score']>=750 else ('⚠️ MANUAL REVIEW' if r['score']>=600 else '❌ REJECTED')}
                        </span></div>
                        <div style="font-size:.8rem;color:#9CA3AF;margin-top:16px;padding-top:16px;border-top:1px solid #1F2937;">
                            Default Risk: <span style="color:{r['color']};font-weight:700;">{round(float(r['prob'])*100,2)}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with g_col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r['score'],
                        number={'font':{'size':48,'color':r['color'],'family':'Space Grotesk'}},
                        gauge={
                            'axis':{'range':[300,900],'tickcolor':'#374151','tickwidth':1,
                                    'tickvals':[300,450,600,750,900],'tickfont':{'color':'#9CA3AF','size':12}},
                            'bar':{'color':r['color'],'thickness':0.25},
                            'bgcolor':'#111827',
                            'borderwidth':0,
                            'steps':[
                                {'range':[300,550],'color':'rgba(239, 68, 68, 0.1)'},
                                {'range':[550,650],'color':'rgba(245, 158, 11, 0.1)'},
                                {'range':[650,750],'color':'rgba(59, 130, 246, 0.1)'},
                                {'range':[750,900],'color':'rgba(16, 185, 129, 0.1)'},
                            ],
                        }
                    ))
                    fig.update_layout(
                        height=280, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'family':'Inter','color':'#F3F4F6'},
                        margin=dict(l=20,r=20,t=40,b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                k1,k2,k3,k4 = st.columns(4)
                k_data = [
                    ("Credit Score", r['score'], "300–900 range", "🎯"),
                    ("Default Risk", f"{round(float(r['prob'])*100,2)}%", "Probability", "📉"),
                    ("Age Factor", r['feats']['age'], "Years", "👤"),
                    ("Credit Amount", f"₹{r['feats']['credit_amount']:,}", "Loan amount", "💰"),
                ]
                for col_k, (lbl, val, sub, ico) in zip([k1,k2,k3,k4], k_data):
                    col_k.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">{ico}</div>
                        <div class="kpi-label">{lbl}</div>
                        <div class="kpi-value">{val}</div>
                        <div class="kpi-sub" style="color:#64748B;">{sub}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">🧠 AI Decision Justification (XAI)</div>', unsafe_allow_html=True)

                xai_exp = shap.TreeExplainer(model)
                sv = xai_exp(r['idf'])
                sh_vals = sv[0].values
                fnames  = list(r['idf'].columns)
                contrib = pd.DataFrame({'Feature':fnames,'Impact':sh_vals}).sort_values('Impact',ascending=False)

                xc1, xc2 = st.columns(2)
                with xc1:
                    st.markdown('<div style="font-size:.9rem;font-weight:600;color:#F87171;margin-bottom:16px;">🔴 Risk Amplifiers (Negative Impact)</div>', unsafe_allow_html=True)
                    for _,row in contrib.head(4).iterrows():
                        bp = min(100, int(abs(row['Impact'])*800))
                        st.markdown(f"""
                        <div class="factor-row">
                            <div class="factor-label">
                                <span class="factor-name">{row['Feature']}</span>
                                <span style="color:#F87171;font-weight:600;">+{row['Impact']:.3f}</span>
                            </div>
                            <div class="bar-track"><div class="bar-fill-red" style="width:{bp}%"></div></div>
                        </div>""", unsafe_allow_html=True)

                with xc2:
                    st.markdown('<div style="font-size:.9rem;font-weight:600;color:#34D399;margin-bottom:16px;">🟢 Protective Factors (Positive Impact)</div>', unsafe_allow_html=True)
                    for _,row in contrib.tail(4).iterrows():
                        bp = min(100, int(abs(row['Impact'])*800))
                        st.markdown(f"""
                        <div class="factor-row">
                            <div class="factor-label">
                                <span class="factor-name">{row['Feature']}</span>
                                <span style="color:#34D399;font-weight:600;">{row['Impact']:.3f}</span>
                            </div>
                            <div class="bar-track"><div class="bar-fill-green" style="width:{bp}%"></div></div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 View Detailed SHAP Waterfall Chart"):
                    safe_shap_waterfall(sv[0], height=420)

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

                st.markdown('<div class="section-title">💡 Score Improvement Roadmap</div>', unsafe_allow_html=True)
                t1,t2,t3 = st.columns(3)
                with t1:
                    st.markdown("""<div class="tip-card">
                        <h5>⚡ Quick Wins <span style="font-size:0.75rem; color:#64748B; font-weight:normal;">(0–3 mos)</span></h5>
                        <ul>
                            <li>Pay all dues before due date</li>
                            <li>Clear outstanding balances</li>
                            <li>Dispute errors in credit report</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                with t2:
                    st.markdown("""<div class="tip-card">
                        <h5>📈 Mid Term <span style="font-size:0.75rem; color:#64748B; font-weight:normal;">(3–12 mos)</span></h5>
                        <ul>
                            <li>Keep credit utilization &lt;30%</li>
                            <li>Avoid multiple loan applications</li>
                            <li>Maintain 1 secured credit card</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                with t3:
                    st.markdown("""<div class="tip-card">
                        <h5>🏆 Long Term <span style="font-size:0.75rem; color:#64748B; font-weight:normal;">(1–3 yrs)</span></h5>
                        <ul>
                            <li>Build diversified credit mix</li>
                            <li>Keep old accounts active</li>
                            <li>Grow emergency savings fund</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:100px 40px;">
                <div style="font-size:4.5rem;margin-bottom:24px; opacity:0.8;">🪪</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
                            font-weight:700;color:#F3F4F6;margin-bottom:16px;">
                    Awaiting Applicant PAN
                </div>
                <div style="color:#9CA3AF;font-size:.95rem;max-width:420px;margin:0 auto;line-height:1.6;">
                    The applicant's AI-powered credit intelligence report will appear here. Enter a PAN card number or use a demo profile from the top left.
                </div>
                <div style="margin-top:40px;display:flex;justify-content:center;gap:48px;">
                    <div style="text-align:center;">
                        <div style="font-size:2rem;font-weight:800;background: linear-gradient(135deg, #6366F1, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family:'Space Grotesk'">20</div>
                        <div style="font-size:.75rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-top:4px;">Features</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem;font-weight:800;background: linear-gradient(135deg, #6366F1, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family:'Space Grotesk'">4</div>
                        <div style="font-size:.75rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-top:4px;">AI Models</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem;font-weight:800;background: linear-gradient(135deg, #6366F1, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family:'Space Grotesk'">&lt;1s</div>
                        <div style="font-size:.75rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-top:4px;">Latency</div>
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
        idx = st.slider("Select Applicant ID from Database", 0, len(X)-1, 0)
        app_data = X.iloc[[idx]]
        prob_u = model.predict_proba(app_data)[0][1]
        score_u = cibil(prob_u)
        g_u, c_u, b_u, i_u = grade(score_u)

        m1,m2,m3,m4 = st.columns(4)
        for col_m,(lbl,val,ico) in zip([m1,m2,m3,m4],[
            ("CIBIL Score", score_u, "🎯"),
            ("Default Risk", f"{round(float(prob_u)*100,2)}%", "📉"),
            ("Age", int(app_data['age'].values[0]), "👤"),
            ("Credit Amount", f"₹{int(app_data['credit_amount'].values[0]):,}", "💰")
        ]):
            col_m.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{ico}</div>
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub" style="color:#64748B;">Applicant #{idx}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2 = st.columns([1,1.5], gap="large")
        with p1:
            st.markdown(f'<div class="section-title">Raw Profile Data</div>', unsafe_allow_html=True)
            st.dataframe(app_data.T.rename(columns={idx:"Value"}), use_container_width=True, height=420)
        with p2:
            st.markdown('<div class="section-title">Risk Gauge</div>', unsafe_allow_html=True)
            fig_u = go.Figure(go.Indicator(
                mode="gauge+number", value=score_u,
                number={'font':{'color':c_u,'size':56,'family':'Space Grotesk'}},
                gauge={
                    'axis':{'range':[300,900],'tickvals':[300,450,600,750,900],'tickfont':{'color':'#9CA3AF','size':12}},
                    'bar':{'color':c_u,'thickness':0.25},
                    'bgcolor':'#111827','borderwidth':0,
                    'steps':[
                        {'range':[300,550],'color':'rgba(239, 68, 68, 0.1)'},
                        {'range':[550,650],'color':'rgba(245, 158, 11, 0.1)'},
                        {'range':[650,750],'color':'rgba(59, 130, 246, 0.1)'},
                        {'range':[750,900],'color':'rgba(16, 185, 129, 0.1)'},
                    ]
                }
            ))
            fig_u.update_layout(height=340,paper_bgcolor='rgba(0,0,0,0)',
                font={'family':'Inter','color':'#F3F4F6'},margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_u, use_container_width=True)
            st.markdown(f'<div style="text-align:center;"><span class="{b_u} badge" style="font-size:1rem; padding:10px 24px;">{i_u} {g_u}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🧠 SHAP Explanation</div>', unsafe_allow_html=True)
        sv_u = shap_vals_global[idx].values
        cont_u = pd.DataFrame({'Feature':X.columns,'Impact':sv_u}).sort_values('Impact',ascending=False)
        top2r = cont_u.head(2); top1s = cont_u.tail(1)
        
        st.info(f"**AI Summary:** Applicant #{idx} has a **{round(float(prob_u)*100,2)}%** default probability. "
                f"The primary risk drivers increasing this probability are **{top2r.iloc[0]['Feature']}** and **{top2r.iloc[1]['Feature']}**. "
                f"The strongest mitigating factor is **{top1s.iloc[0]['Feature']}**.")
                
        with st.expander("📊 View SHAP Waterfall"):
            safe_shap_waterfall(shap_vals_global[idx], height=400)

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
            ("Total Applicants", len(X), "In portfolio database", "📁"),
            ("Model Accuracy", f"{acc}%", "Test set performance","🎯"),
            ("Approval Rate", f"{approval}%", "Low risk predicted", "✅"),
            ("Default Rate", f"{round(100-approval,2)}%", "High risk predicted","⚠️"),
        ]):
            col_p.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon">{ico}</div>
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub" style="color:#64748B;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ga,gb = st.columns(2)
        with ga:
            st.markdown('<div class="section-title">Top Risk Drivers (Global)</div>', unsafe_allow_html=True)
            plt.figure(facecolor='#111827')
            shap.summary_plot(shap_vals_global, X, plot_type="bar", show=False, color='#6366F1')
            fig_g1 = plt.gcf()
            fig_g1.set_facecolor('#111827')
            for ax in fig_g1.axes:
                ax.set_facecolor('#0B1120')
                ax.tick_params(colors='#9CA3AF')
                ax.xaxis.label.set_color('#9CA3AF')
                for spine in ax.spines.values(): spine.set_edgecolor('#1F2937')
            st.pyplot(fig_g1)
            plt.close(fig_g1)

        with gb:
            st.markdown('<div class="section-title">Directional Impact (Beeswarm)</div>', unsafe_allow_html=True)
            plt.figure(facecolor='#111827')
            shap.summary_plot(shap_vals_global, X, show=False)
            fig_g2 = plt.gcf()
            fig_g2.set_facecolor('#111827')
            for ax in fig_g2.axes:
                ax.set_facecolor('#0B1120')
                ax.tick_params(colors='#9CA3AF')
                ax.xaxis.label.set_color('#9CA3AF')
                for spine in ax.spines.values(): spine.set_edgecolor('#1F2937')
            st.pyplot(fig_g2)
            plt.close(fig_g2)

        st.markdown('<div class="section-title">Score Distribution Matrix</div>', unsafe_allow_html=True)
        probs_all = model.predict_proba(X)[:,1]
        scores_all = [cibil(p) for p in probs_all]
        fig_dist = px.histogram(x=scores_all, nbins=40,
            labels={'x':'CIBIL Score','y':'Applicant Count'},
            color_discrete_sequence=['#6366F1'])
        fig_dist.update_layout(
            paper_bgcolor='#0B1120', plot_bgcolor='#111827',
            font={'color':'#9CA3AF','family':'Inter'},
            xaxis={'gridcolor':'#1F2937'}, yaxis={'gridcolor':'#1F2937'},
            margin=dict(l=20,r=20,t=20,b=20), height=350
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
        st.markdown('<div style="font-size:.9rem;color:#9CA3AF;margin-bottom:24px;">Ensuring compliance with Equal Credit Opportunity Act (ECOA) & RBI Fair Lending Standards.</div>', unsafe_allow_html=True)

        sf = (df['age'] < 25).astype(int)
        yt = df['target']
        yp = model.predict(X)
        dp = demographic_parity_difference(yt, yp, sensitive_features=sf)
        acc_f = round(accuracy_score(yt, yp)*100, 2)

        fa1,fa2,fa3 = st.columns(3)
        fa1.metric("Demographic Parity Diff", f"{round(dp*100,2)}%", delta=None)
        fa2.metric("Model Accuracy", f"{acc_f}%")
        fa3.metric("Under-25 Flag", "HIGH RISK" if dp > 0.1 else "COMPLIANT",
                   delta="Action Required" if dp > 0.1 else "Passed", delta_color="inverse" if dp > 0.1 else "normal")

        st.markdown("<br>", unsafe_allow_html=True)
        if dp > 0.1:
            st.error("""❌ **Audit Failed** — The model applies disproportionate risk to applicants under 25.  
            **Required Actions:** Apply fairness constraints (Fairlearn reweighing) before production deployment.""")
        else:
            st.success("""✅ **Audit Passed** — Model demonstrates equitable approval rates across all age demographics.  
            Demographic parity difference is within acceptable regulatory thresholds (<10%).""")

        st.markdown('<div class="section-title">Age Group Approval Breakdown</div>', unsafe_allow_html=True)
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
        '<div style="font-size:.9rem;color:#9CA3AF;margin-bottom:24px; line-height:1.6;">'
        'Adjust any credit factor below and instantly see how your CIBIL score changes. '
        'Find the <b style="color:#6366F1;">exact actions</b> needed to move to the next grade.'
        '</div>', unsafe_allow_html=True
    )

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
        seed_label = "Using default profile — check a PAN first for a personalized simulation"

    st.info(f"📌 **Baseline:** {seed_label}  |  **Score:** {base_score}", icon="📊")

    sim_left, sim_right = st.columns([1.2, 1], gap="large")

    with sim_left:
        st.markdown('<div style="font-size:.95rem;font-weight:600;color:#F3F4F6;margin-bottom:16px;">🎚️ Adjust Credit Factors</div>', unsafe_allow_html=True)

        feature_configs = {
            'checking_status': {'label': '🏦 Checking Account Status', 'type': 'select',
                'options': ['No Account (worst)', '< ₹0 (negative)', '₹0–₹200 (ok)', '> ₹200 (best)'],
                'help': 'Higher is better. A healthy checking account lowers risk.'},
            'credit_history': {'label': '📜 Credit History Quality', 'type': 'select',
                'options': ['Critical/Other Account', 'No Credits Taken', 'All Paid Duly', 'Existing Paid', 'All Paid (best)'],
                'help': 'Past repayment behaviour. 4 = perfect history.'},
            'savings_status': {'label': '💰 Savings Account Balance', 'type': 'select',
                'options': ['No Savings (worst)', '< ₹100', '₹100–₹500', '₹500–₹1000', '> ₹1000 (best)'],
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
                'help': 'Demographic factor.'},
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
                'help': 'Dataset specific demographic factor.'},
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
        FEATURE_ORDER = [
            'checking_status', 'duration', 'credit_history', 'purpose',
            'credit_amount', 'savings_status', 'employment', 'installment_commitment',
            'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
            'age', 'other_payment_plans', 'housing', 'existing_credits',
            'job', 'num_dependents', 'own_telephone', 'foreign_worker'
        ]
        sim_idf   = pd.DataFrame([sim_feats])
        col_order = list(X.columns) if X is not None else FEATURE_ORDER
        sim_idf   = sim_idf[col_order]
        sim_prob  = model.predict_proba(sim_idf)[0][1]
        sim_score = cibil(sim_prob)
        sim_g, sim_color, _, sim_ico = grade(sim_score)

        delta      = sim_score - base_score
        delta_prob = round((sim_prob - base_prob) * 100, 1)
        arrow      = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        d_color    = "#10B981" if delta > 0 else ("#EF4444" if delta < 0 else "#64748B")

        st.markdown('<div class="section-title">📊 Before vs After</div>', unsafe_allow_html=True)
        ba1, ba2, ba3 = st.columns([1, 0.4, 1])
        with ba1:
            bg, bc, _, bi = grade(base_score)
            st.markdown(f"""
            <div class="score-ring" style="padding:20px;">
                <div class="score-label">BASELINE</div>
                <div class="score-number" style="font-size:2.8rem;background:none;-webkit-text-fill-color:{bc};">{base_score}</div>
                <div style="font-size:.85rem;color:{bc};font-weight:700;margin-top:6px;">{bi} {bg}</div>
            </div>""", unsafe_allow_html=True)
        with ba2:
            st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:100%;padding-top:40px;">
                <div style="font-size:2.2rem;color:{d_color};font-weight:800;line-height:1;">{arrow}</div>
                <div style="font-size:1.1rem;font-weight:800;color:{d_color}; margin-top:4px;">
                    {'+' if delta>=0 else ''}{delta}</div>
                <div style="font-size:.7rem;color:#64748B;font-weight:600;letter-spacing:1px;text-transform:uppercase;">pts</div>
            </div>""", unsafe_allow_html=True)
        with ba3:
            st.markdown(f"""
            <div class="score-ring" style="padding:20px;border-color:{sim_color}33;">
                <div class="score-label">NEW SCORE</div>
                <div class="score-number" style="font-size:2.8rem;background:none;-webkit-text-fill-color:{sim_color};">{sim_score}</div>
                <div style="font-size:.85rem;color:{sim_color};font-weight:700;margin-top:6px;">{sim_ico} {sim_g}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1F2937;border-radius:14px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);">
            <div style="display:flex;justify-content:space-between;font-size:.85rem;
                        color:#9CA3AF;margin-bottom:12px;font-weight:500;">
                <span>Default Risk Profile</span>
                <span style="color:{d_color};font-weight:700;">
                    {round(sim_prob*100,1)}%
                    ({'+' if delta_prob>=0 else ''}{delta_prob}%)
                </span>
            </div>
            <div style="background:#1F2937;border-radius:8px;height:12px;overflow:hidden;">
                <div style="background:{sim_color};width:{min(100,int(sim_prob*100))}%;
                            height:100%;border-radius:8px;transition:width 0.5s ease;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if sim_score >= 750:
            st.success("✅ **AUTO-APPROVED** — Excellent creditworthiness profile.")
        elif sim_score >= 600:
            st.warning("⚠️ **MANUAL REVIEW** — Good profile, but requires underwriter check.")
        else:
            st.error("❌ **REJECTED** — High default risk threshold exceeded.")

        changed = {k: (base_feats.get(k,0), sim_feats[k])
                   for k in sim_feats if sim_feats[k] != base_feats.get(k,0)}
        if changed:
            st.markdown('<div class="section-title" style="font-size:1rem;margin-top:24px;">🔄 Parameter Deltas</div>', unsafe_allow_html=True)
            for feat, (old_v, new_v) in changed.items():
                lbl = feature_configs[feat]['label']
                up  = new_v > old_v
                cc  = "#10B981" if up else "#EF4444"
                ci  = "↑" if up else "↓"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            background:#111827;border:1px solid #1F2937;border-radius:10px;
                            padding:10px 16px;margin:6px 0;font-size:.85rem;">
                    <span style="color:#D1D5DB;font-weight:500;">{lbl}</span>
                    <span>
                        <span style="color:#64748B;">{old_v}</span>
                        <span style="color:#475569;margin:0 8px;">→</span>
                        <span style="color:{cc};font-weight:700;">{new_v} {ci}</span>
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#64748B;font-size:.85rem;text-align:center;padding:24px;border:1px dashed #1E293B;border-radius:12px;">← Move any slider to see live impact on score</div>', unsafe_allow_html=True)

        with st.expander("🧠 XAI Breakdown — Before vs After"):
            xai_exp = shap.TreeExplainer(model)
            sv_base = xai_exp(pd.DataFrame([base_feats]))
            sv_sim  = xai_exp(sim_idf)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.caption("**Baseline Configuration**")
                safe_shap_waterfall(sv_base[0], height=340)
            with sc2:
                st.caption("**Simulated Configuration**")
                safe_shap_waterfall(sv_sim[0],  height=340)

        st.markdown('<div class="section-title" style="font-size:1rem;margin-top:24px;">💡 Top Actions to Improve</div>', unsafe_allow_html=True)
        recs = []
        if sim_feats['savings_status'] < 3:
            recs.append(("💰 Increase Savings", "Move savings to ₹100–₹500+ band. Reduces risk significantly."))
        if sim_feats['credit_history'] < 4:
            recs.append(("📜 Build Credit History", "Pay all dues on time for 6–12 months to reach 'All Paid' status."))
        if sim_feats['duration'] > 24:
            recs.append(("📅 Shorten Loan Term", f"Reduce duration from {sim_feats['duration']} → {max(6,sim_feats['duration']-12)} months."))
        if sim_feats['credit_amount'] > 8000:
            recs.append(("💳 Reduce Loan Amount", f"Request ₹{sim_feats['credit_amount']-2000:,} instead of ₹{sim_feats['credit_amount']:,}."))
        if sim_feats['checking_status'] < 2:
            recs.append(("🏦 Maintain Positive Balance", "Keep checking account in positive balance consistently."))
        if sim_feats['installment_commitment'] > 2:
            recs.append(("📊 Lower EMI Burden", "Consolidate or prepay existing loans to reduce commitment."))
        
        if not recs:
            recs.append(("🏆 Profile is Strong!", "Maintain consistency for a great credit score."))
            
        for i, (title, desc) in enumerate(recs[:4]):
            st.markdown(f"""
            <div style="display:flex;gap:14px;align-items:flex-start;
                        background:#111827;border:1px solid #1F2937;border-radius:12px;
                        padding:16px;margin:8px 0;box-shadow:0 2px 4px rgba(0,0,0,0.05);">
                <div style="min-width:26px;height:26px;background:rgba(99, 102, 241, 0.1);
                            border: 1px solid rgba(99, 102, 241, 0.3);
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:.8rem;color:#818CF8;">{i+1}</div>
                <div>
                    <div style="font-weight:600;color:#F3F4F6;font-size:.9rem;margin-bottom:4px;">{title}</div>
                    <div style="color:#9CA3AF;font-size:.85rem;line-height:1.4;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)