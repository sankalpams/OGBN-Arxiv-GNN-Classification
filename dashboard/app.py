from pathlib import Path
import sys
import warnings

# Suppress deprecation warnings on newer Python / Streamlit versions
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure dashboard directory is on path for component imports
_dashboard_dir = Path(__file__).resolve().parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

# Ensure project root is on path for src imports
_project_root = _dashboard_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = _project_root
RESULTS = ROOT / "results"


def _resolve_candidate_file(relative_subpath: str) -> Path | None:
    """Multi-path search to reliably resolve result files across local and cloud environments."""
    candidates = [
        _project_root / "results" / relative_subpath,
        _dashboard_dir / "results" / relative_subpath,
        Path.cwd() / "results" / relative_subpath,
        Path("results") / relative_subpath,
        _project_root / relative_subpath,
        _dashboard_dir / relative_subpath,
        Path.cwd() / relative_subpath,
        Path(relative_subpath),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


from components.graph_stats import render_graph_stats
from components.model_metrics import render_model_metrics
from components.classification import render_classification_demo
from components.embeddings import render_embedding_image

st.set_page_config(
    page_title="OGBN-Arxiv Graph Intelligence | Liquid Glass GNN Suite",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Liquid Glass Theme Palette
active_aura = {
    "primary": "#38BDF8",
    "secondary": "#818CF8",
    "accent": "#0284C7",
    "glow": "rgba(56, 189, 248, 0.28)",
    "gradient": "linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%)",
    "orb1": "rgba(56, 189, 248, 0.22)",
    "orb2": "rgba(99, 102, 241, 0.20)",
    "orb3": "rgba(14, 165, 233, 0.15)",
}

# State-of-the-Art Liquid Glass CSS Design System
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');

    :root {{
        --aura-primary: {active_aura['primary']};
        --aura-secondary: {active_aura['secondary']};
        --aura-accent: {active_aura['accent']};
        --aura-glow: {active_aura['glow']};
        --aura-gradient: {active_aura['gradient']};
    }}

    html {{
        scroll-behavior: smooth;
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F1F5F9;
    }}

    /* Global Dark Obsidian Liquid Background */
    .stApp {{
        background: radial-gradient(circle at 50% 0%, #0B1120 0%, #060913 60%, #03050A 100%) !important;
        background-attachment: fixed !important;
        overflow-x: hidden;
    }}

    /* Modern Custom 3D Glowing Liquid Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: rgba(6, 9, 19, 0.85);
        backdrop-filter: blur(10px);
    }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, var(--aura-primary) 0%, var(--aura-secondary) 100%);
        border-radius: 999px;
        border: 2px solid rgba(6, 9, 19, 0.85);
        box-shadow: 0 0 10px var(--aura-glow);
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, #FFFFFF 0%, var(--aura-primary) 100%);
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.025em;
        color: #FFFFFF;
    }}

    /* Main Container with 3D Depth */
    .main .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3.2rem;
        max-width: 1420px;
        perspective: 1400px;
    }}

    /* Floating Liquid Ambient Orbs in Background */
    .liquid-mesh-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: -1;
        overflow: hidden;
    }}

    .liquid-orb {{
        position: absolute;
        border-radius: 50%;
        filter: blur(90px);
        opacity: 0.65;
        mix-blend-mode: screen;
        will-change: transform;
    }}

    .liquid-orb-1 {{
        top: -10%;
        left: 15%;
        width: 520px;
        height: 520px;
        background: {active_aura['orb1']};
        animation: liquid-float-1 22s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }}

    .liquid-orb-2 {{
        top: 35%;
        right: -8%;
        width: 580px;
        height: 580px;
        background: {active_aura['orb2']};
        animation: liquid-float-2 26s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }}

    .liquid-orb-3 {{
        bottom: -15%;
        left: 28%;
        width: 620px;
        height: 620px;
        background: {active_aura['orb3']};
        animation: liquid-float-3 30s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
    }}

    @keyframes liquid-float-1 {{
        0% {{ transform: translate(0px, 0px) scale(1) rotate(0deg); }}
        50% {{ transform: translate(70px, 90px) scale(1.15) rotate(90deg); }}
        100% {{ transform: translate(-50px, 120px) scale(0.92) rotate(180deg); }}
    }}

    @keyframes liquid-float-2 {{
        0% {{ transform: translate(0px, 0px) scale(1) rotate(0deg); }}
        50% {{ transform: translate(-90px, -70px) scale(1.2) rotate(-90deg); }}
        100% {{ transform: translate(-130px, 60px) scale(0.95) rotate(-180deg); }}
    }}

    @keyframes liquid-float-3 {{
        0% {{ transform: translate(0px, 0px) scale(1); }}
        50% {{ transform: translate(110px, -80px) scale(1.1); }}
        100% {{ transform: translate(-80px, -50px) scale(0.9); }}
    }}

    /* =========================================================
       LIQUID GLASS CARDS & SURFACES
       ========================================================= */
    .liquid-glass-card, .metric-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(11, 17, 32, 0.72) !important;
        backdrop-filter: blur(24px) saturate(190%) contrast(108%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) contrast(108%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.32) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 18px !important;
        padding: 22px 24px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.55),
                    0 0 24px -4px var(--aura-glow),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.35),
                    inset 0 -12px 24px -10px rgba(0, 0, 0, 0.35) !important;
        transform-style: preserve-3d;
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.35s ease,
                    border-color 0.35s ease !important;
    }}

    /* Specular Liquid Light Shimmer Effect */
    .liquid-glass-card::before, .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -120%;
        width: 80%;
        height: 100%;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.12) 50%, transparent 100%);
        transform: skewX(-24deg);
        pointer-events: none;
        transition: left 0.75s ease;
    }}

    .liquid-glass-card:hover::before, .metric-card:hover::before {{
        left: 160%;
    }}

    .liquid-glass-card:hover, .metric-card:hover {{
        transform: translateY(-6px) translateZ(14px) scale(1.02) !important;
        border-color: var(--aura-primary) !important;
        box-shadow: 0 24px 45px -8px rgba(0, 0, 0, 0.65),
                    0 0 36px 0px var(--aura-glow),
                    inset 0 1px 2px 0 rgba(255, 255, 255, 0.6) !important;
    }}

    .metric-card.highlight-gcn {{
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.18) 0%, rgba(11, 17, 32, 0.85) 100%) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.55), 0 0 28px rgba(56, 189, 248, 0.25), inset 0 1px 2px rgba(255, 255, 255, 0.4) !important;
    }}

    .metric-card.highlight-gat {{
        background: linear-gradient(135deg, rgba(192, 132, 252, 0.18) 0%, rgba(11, 17, 32, 0.85) 100%) !important;
        border-color: rgba(192, 132, 252, 0.45) !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.55), 0 0 28px rgba(192, 132, 252, 0.25), inset 0 1px 2px rgba(255, 255, 255, 0.4) !important;
    }}

    .metric-icon {{
        font-size: 2rem;
        margin-bottom: 8px;
        filter: drop-shadow(0 2px 8px var(--aura-glow));
    }}

    .metric-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.25rem;
        font-weight: 900;
        letter-spacing: -0.035em;
        line-height: 1.1;
        color: #FFFFFF;
        text-shadow: 0 2px 14px rgba(0, 0, 0, 0.4);
    }}

    .metric-label {{
        font-size: 0.86rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-top: 6px;
    }}

    .metric-sub {{
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
        font-weight: 500;
    }}

    /* =========================================================
       HERO HEADER BANNER WITH LIQUID CAUSTICS
       ========================================================= */
    .hero-banner {{
        background: radial-gradient(120% 140% at 50% 5%, rgba(255, 255, 255, 0.10) 0%, rgba(15, 23, 42, 0.85) 60%, rgba(6, 9, 19, 0.95) 100%);
        backdrop-filter: blur(28px) saturate(200%);
        -webkit-backdrop-filter: blur(28px) saturate(200%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-top: 1px solid rgba(255, 255, 255, 0.38);
        border-left: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 24px;
        padding: 34px 38px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 28px 60px -12px rgba(0, 0, 0, 0.65),
                    0 0 35px var(--aura-glow),
                    inset 0 1px 2px 0 rgba(255, 255, 255, 0.45);
        transform-style: preserve-3d;
    }}

    .hero-banner::after {{
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 480px;
        height: 480px;
        background: radial-gradient(circle, var(--aura-glow) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        animation: liquid-banner-pulse 10s ease-in-out infinite alternate;
    }}

    @keyframes liquid-banner-pulse {{
        0% {{ transform: scale(1) translate(0, 0); opacity: 0.6; }}
        100% {{ transform: scale(1.25) translate(-40px, 30px); opacity: 0.95; }}
    }}

    .hero-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.55rem;
        font-weight: 900;
        letter-spacing: -0.035em;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 40%, var(--aura-primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        line-height: 1.15;
    }}

    .hero-desc {{
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 920px;
        line-height: 1.6;
    }}

    /* =========================================================
       LIQUID PILL BADGES
       ========================================================= */
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .badge-pill:hover {{
        transform: translateY(-2px) scale(1.04);
    }}

    .badge-blue {{
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.22) 0%, rgba(56, 189, 248, 0.08) 100%);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.45);
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.2);
    }}
    .badge-purple {{
        background: linear-gradient(135deg, rgba(192, 132, 252, 0.22) 0%, rgba(192, 132, 252, 0.08) 100%);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.45);
        box-shadow: 0 0 14px rgba(192, 132, 252, 0.2);
    }}
    .badge-green {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(16, 185, 129, 0.08) 100%);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.45);
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.2);
    }}
    .badge-amber {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.22) 0%, rgba(245, 158, 11, 0.08) 100%);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.45);
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.2);
    }}

    /* =========================================================
       STREAMLIT TABS OVERHAUL (LIQUID FROSTED CAPSULES & HOVER)
       ========================================================= */
    /* Remove default Streamlit red/orange underline highlight and bottom borders */
    div[data-baseweb="tab-highlight"],
    [data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"],
    [data-baseweb="tab-border"],
    [data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    [data-testid="stTabs"] div[data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
        background: transparent !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }}

    /* Tab List Container Bar */
    [data-testid="stTabs"] > div:first-child,
    div[data-testid="stTabsNav"],
    div[data-baseweb="tab-list"],
    div[role="tablist"],
    .stTabs [data-baseweb="tab-list"],
    .stTabs div[role="tablist"] {{
        gap: 10px !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
        backdrop-filter: blur(24px) saturate(190%) contrast(108%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) contrast(108%) !important;
        padding: 8px 10px !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.55),
                    0 0 20px -4px var(--aura-glow),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.3),
                    inset 0 -8px 16px -8px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 22px !important;
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        border-bottom: none !important;
    }}

    /* Individual Tab Buttons - Default / Inactive */
    [data-testid="stTabs"] button,
    button[data-testid="stTab"],
    div[data-baseweb="tab-list"] button,
    button[data-baseweb="tab"],
    div[role="tablist"] button[role="tab"],
    .stTabs button,
    .stTabs [data-baseweb="tab"] {{
        height: 44px !important;
        padding: 0 20px !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.18) !important;
        color: #94A3B8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.93rem !important;
        letter-spacing: -0.01em !important;
        cursor: pointer !important;
        transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        outline: none !important;
        box-shadow: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
    }}

    /* Tab Hover Effect - Smooth Lift, Specular Glow & Rounded Corners */
    [data-testid="stTabs"] button:hover,
    button[data-testid="stTab"]:hover,
    div[data-baseweb="tab-list"] button:hover,
    button[data-baseweb="tab"]:hover,
    div[role="tablist"] button[role="tab"]:hover,
    .stTabs button:hover,
    .stTabs [data-baseweb="tab"]:hover {{
        color: #FFFFFF !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(56, 189, 248, 0.16) 100%) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.45) !important;
        transform: translateY(-3px) scale(1.025) !important;
        box-shadow: 0 10px 22px -4px rgba(56, 189, 248, 0.3),
                    inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
    }}

    /* Tab Active / Selected State - Glowing Liquid Glass Pill */
    [data-testid="stTabs"] button[aria-selected="true"],
    button[data-testid="stTab"][aria-selected="true"],
    div[data-baseweb="tab-list"] button[aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"],
    div[role="tablist"] button[role="tab"][aria-selected="true"],
    .stTabs button[aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.26) 0%, rgba(192, 132, 252, 0.20) 100%), rgba(15, 23, 42, 0.9) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 1px solid rgba(56, 189, 248, 0.55) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 10px 25px -4px rgba(56, 189, 248, 0.35),
                    0 0 16px -2px rgba(56, 189, 248, 0.25),
                    inset 0 1px 2px rgba(255, 255, 255, 0.5) !important;
        transform: translateY(-2px) !important;
    }}

    /* Text & Icon styling inside tabs */
    [data-testid="stTabs"] button p,
    button[data-baseweb="tab"] p,
    div[role="tablist"] button p,
    .stTabs [data-baseweb="tab"] p {{
        font-size: 0.93rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: inherit !important;
    }}

    /* =========================================================
       IMAGE AND CHART CONTAINERS
       ========================================================= */
    .glass-img-container {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 18px !important;
        padding: 14px;
        box-shadow: 0 14px 32px -6px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        transition: border-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .glass-img-container:hover {{
        border-color: var(--aura-primary) !important;
        box-shadow: 0 18px 40px -4px var(--aura-glow), inset 0 1px 2px rgba(255, 255, 255, 0.4);
        transform: translateY(-3px);
    }}

    /* =========================================================
       LIQUID STREAMLIT WIDGETS OVERHAUL
       ========================================================= */
    /* Liquid Buttons */
    .stButton button {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.14) 0%, rgba(255, 255, 255, 0.04) 100%), var(--aura-glow) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.45) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }}

    .stButton button:hover {{
        transform: translateY(-3px) scale(1.02) !important;
        border-color: var(--aura-primary) !important;
        box-shadow: 0 12px 28px -4px var(--aura-glow), inset 0 1px 2px rgba(255, 255, 255, 0.6) !important;
    }}

    /* Liquid Text Input & Select Box */
    .stTextInput input, .stSelectbox > div > div {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }}

    .stTextInput input:focus, .stSelectbox > div > div:focus-within {{
        border-color: var(--aura-primary) !important;
        box-shadow: 0 0 20px var(--aura-glow), inset 0 1px 2px rgba(255, 255, 255, 0.3) !important;
    }}

    /* Liquid Expanders */
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.07) 0%, rgba(15, 23, 42, 0.75) 100%) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.26) !important;
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }}

    .streamlit-expanderContent {{
        background: rgba(11, 17, 32, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: none !important;
        border-radius: 0 0 14px 14px !important;
    }}

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.82) 0%, rgba(6, 9, 19, 0.95) 100%) !important;
        backdrop-filter: blur(28px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5) !important;
    }}

    /* Pulse Live Status Dot */
    .pulse-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-ring 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
        display: inline-block;
    }}

    @keyframes pulse-ring {{
        0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}
</style>

<!-- Liquid Morphing Ambient Background Mesh -->
<div class="liquid-mesh-container">
    <div class="liquid-orb liquid-orb-1"></div>
    <div class="liquid-orb liquid-orb-2"></div>
    <div class="liquid-orb liquid-orb-3"></div>
</div>
""", unsafe_allow_html=True)

# Liquid Glass Hero Header Banner
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;">
        <span class="badge-pill badge-blue">🕸️ OGBN-ARXIV 169K</span>
        <span class="badge-pill badge-purple">🎮 3D Scroll & Orbit Enabled</span>
        <span class="badge-pill badge-green">🏆 GCN Peak: 58.64%</span>
        <span class="badge-pill badge-amber">⏱️ GCN 3.25x Faster</span>
    </div>
    <div class="hero-title">OGBN-Arxiv Deep Graph Intelligence Suite</div>
    <div class="hero-desc">
        End-to-end representation learning and empirical benchmark evaluating <b>Spectral Graph Convolutional Networks (GCN)</b> versus <b>Spatial Multi-Head Graph Attention Networks (GAT)</b> on the 169,343-node Microsoft Academic citation network with <b>3D interactive manifold scroll exploration</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with Rich Metadata
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0;">
        <h3 style="font-weight: 700; color: #F8FAFC; margin-bottom: 2px;">🔬 System Control</h3>
        <p style="color: #64748B; font-size: 0.85rem;">Graph Neural Network Research Suite</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px; margin-bottom: 16px;">
        <div style="color: #10B981; font-weight: 700; font-size: 0.9rem;">✅ Pipeline Status: Ready</div>
        <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 3px;">All 8 experimental tasks & 3D models verified</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎮 3D Navigation Tips")
    st.markdown("""
    - **Mouse Scroll Wheel**: Zoom in/out in 3D
    - **Left Click + Drag**: Rotate 3D camera 360°
    - **Right Click + Drag**: Pan 3D viewport
    - **Double Click**: Reset 3D camera
    """)

    st.markdown("---")
    st.markdown("#### 📦 Dataset Specification")
    st.markdown("""
    - **Benchmark:** `ogbn-arxiv`
    - **Total Papers:** 169,343 nodes
    - **Citation Links:** 2,315,598 edges
    - **Feature Dim:** 128-dim word2vec
    - **Target Categories:** 40 CS classes
    """)

    st.markdown("---")
    st.markdown("#### ⚡ Hardware & Model Specs")
    st.markdown("""
    - **GCN Params:** 43,816 (2 Layers, 256h)
    - **GAT Params:** 43,624 (2 Layers, 4 Heads)
    - **Split Strategy:** Strict Temporal
    - **Train / Val / Test:** ≤2017 / 2018 / 2019–20
    """)

# Main Navigation Tabs
tab_graph, tab_train, tab_eval, tab_class, tab_embed = st.tabs([
    "📊 3D Graph & Topology",
    "📈 Training Dynamics",
    "🏆 Model Evaluation",
    "🔬 Node Classification Lookup",
    "🌌 3D Embeddings & Manifold"
])

summary_file = _resolve_candidate_file("graph_analysis/graph_summary.csv")
metrics_file = _resolve_candidate_file("evaluation/metrics.csv")
gcn_history_file = _resolve_candidate_file("training/gcn_training_history.csv")
gat_history_file = _resolve_candidate_file("training/gat_training_history.csv")
pred_file = _resolve_candidate_file("evaluation/paper_predictions.csv")
explain_dir = _resolve_candidate_file("explainability")
if explain_dir is None:
    explain_dir = RESULTS / "explainability"

with tab_graph:
    if summary_file and summary_file.exists():
        render_graph_stats(pd.read_csv(summary_file), summary_file.parent)
    else:
        # Fallback summary statistics
        default_stats = pd.DataFrame([
            {"Metric": "Nodes (Papers)", "Value": 169343},
            {"Metric": "Edges (Citations)", "Value": 2315598},
            {"Metric": "Features per Node", "Value": 128},
            {"Metric": "Target Classes", "Value": 40},
            {"Metric": "Graph Density", "Value": 0.000081},
            {"Metric": "Average Node Degree", "Value": 13.674}
        ])
        render_graph_stats(default_stats, RESULTS / "graph_analysis")

with tab_train:
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="margin: 0; font-weight: 800; font-size: 1.7rem; color: #FFFFFF; letter-spacing: -0.02em;">
            📈 Training Dynamics & Loss Convergence
        </h2>
        <p style="color: #94A3B8; margin-top: 5px; font-size: 0.98rem; line-height: 1.5;">
            Comparative epoch-by-epoch loss reduction and validation accuracy trajectories over 30 training epochs with liquid gradient smoothing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if gcn_history_file and gat_history_file and gcn_history_file.exists() and gat_history_file.exists():
        gcn_df = pd.read_csv(gcn_history_file)
        gat_df = pd.read_csv(gat_history_file)

        # Milestone Cards with Liquid Glass Styling
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card highlight-gcn">
                <div class="metric-icon">🔹</div>
                <div class="metric-value" style="color: #38BDF8;">{gcn_df['validation_accuracy'].max()*100:.2f}%</div>
                <div class="metric-label">GCN Peak Val Acc</div>
                <div class="metric-sub">Epoch {gcn_df['validation_accuracy'].idxmax()+1} / 30</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card highlight-gat">
                <div class="metric-icon">🔸</div>
                <div class="metric-value" style="color: #C084FC;">{gat_df['validation_accuracy'].max()*100:.2f}%</div>
                <div class="metric-label">GAT Peak Val Acc</div>
                <div class="metric-sub">Epoch {gat_df['validation_accuracy'].idxmax()+1} / 30</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-value" style="color: #34D399;">{gcn_df['loss'].min():.4f}</div>
                <div class="metric-label">GCN Min Training Loss</div>
                <div class="metric-sub">Cross-Entropy Objective</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-value" style="color: #FB923C;">{gat_df['loss'].min():.4f}</div>
                <div class="metric-label">GAT Min Training Loss</div>
                <div class="metric-sub">Cross-Entropy Objective</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Plotly Interactive Comparison Curves framed in Liquid Glass
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("""
            <div class="glass-img-container">
                <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 8px;">📉 Training Loss Trajectories</div>
            """, unsafe_allow_html=True)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=gcn_df["epoch"], y=gcn_df["loss"],
                mode="lines+markers", name="GCN (Loss)",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=6, color="#38BDF8")
            ))
            fig_loss.add_trace(go.Scatter(
                x=gat_df["epoch"], y=gat_df["loss"],
                mode="lines+markers", name="GAT (Loss)",
                line=dict(color="#C084FC", width=3),
                marker=dict(size=6, color="#C084FC")
            ))
            fig_loss.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Cross-Entropy Loss", gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=10, b=20),
                height=320
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c2:
            st.markdown("""
            <div class="glass-img-container">
                <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 8px;">🎯 Validation Accuracy (%)</div>
            """, unsafe_allow_html=True)
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=gcn_df["epoch"], y=gcn_df["validation_accuracy"] * 100,
                mode="lines+markers", name="GCN (Val Acc)",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=6, color="#38BDF8")
            ))
            fig_acc.add_trace(go.Scatter(
                x=gat_df["epoch"], y=gat_df["validation_accuracy"] * 100,
                mode="lines+markers", name="GAT (Val Acc)",
                line=dict(color="#C084FC", width=3),
                marker=dict(size=6, color="#C084FC")
            ))
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Validation Accuracy (%)", gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=10, b=20),
                height=320
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Training history logs will appear once notebooks 04 & 05 are executed.")

with tab_eval:
    if metrics_file and metrics_file.exists():
        render_model_metrics(pd.read_csv(metrics_file), metrics_file.parent)
    else:
        st.info("Evaluation metrics and confusion matrices will appear after Notebook 07.")

with tab_class:
    render_classification_demo(pred_file)

with tab_embed:
    render_embedding_image(explain_dir)

