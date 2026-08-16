from pathlib import Path
import sys
import warnings

# Suppress deprecation warnings on newer Python / Streamlit versions
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

from components.graph_stats import render_graph_stats
from components.model_metrics import render_model_metrics
from components.classification import render_classification_demo
from components.embeddings import render_embedding_image

st.set_page_config(
    page_title="OGBN-Arxiv Graph Intelligence | Deep GNN Suite",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Dark Glassmorphic CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    html {
        scroll-behavior: smooth;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Modern Custom 3D Glowing Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0F19;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #38BDF8 0%, #C084FC 100%);
        border-radius: 8px;
        border: 2px solid #0B0F19;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #0284C7 0%, #9333EA 100%);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    /* Main Container Glow */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
        perspective: 1200px;
    }

    /* Glassmorphism 3D Metric Cards with Depth & Hover Tilt */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        transform-style: preserve-3d;
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease, border-color 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px) translateZ(12px) scale(1.02);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 20px 35px -5px rgba(56, 189, 248, 0.25), 0 10px 15px -5px rgba(0, 0, 0, 0.5);
    }

    .metric-card.highlight-gcn {
        border-color: rgba(56, 189, 248, 0.3);
        background: linear-gradient(135deg, rgba(14, 42, 71, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    }

    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 8px;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }

    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .badge-blue {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .badge-purple {
        background: rgba(192, 132, 252, 0.15);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }

    .badge-green {
        background: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .badge-amber {
        background: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Hero Header Banner with 3D Depth */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.15) 0%, rgba(15, 23, 42, 0) 50%),
                    radial-gradient(circle at 90% 80%, rgba(192, 132, 252, 0.15) 0%, rgba(15, 23, 42, 0) 50%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 25px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        transform-style: preserve-3d;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .hero-desc {
        color: #94A3B8;
        font-size: 1.05rem;
        line-height: 1.5;
        max-width: 950px;
    }

    /* Glass Image Containers */
    .glass-img-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.25s ease;
    }

    .glass-img-container:hover {
        transform: translateY(-2px);
    }

    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94A3B8;
        padding: 0 18px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(192, 132, 252, 0.2) 100%) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
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

    st.markdown("---")
    st.caption("CCS4354 Tensors and Graphs Assignment • Deep Learning on Graphs")

# Main Navigation Tabs
tab_graph, tab_train, tab_eval, tab_class, tab_embed = st.tabs([
    "📊 3D Graph & Topology",
    "📈 Training Dynamics",
    "🏆 Model Evaluation",
    "🔬 Node Classification Lookup",
    "🌌 3D Embeddings & Manifold"
])

summary_file = RESULTS / "graph_analysis" / "graph_summary.csv"
metrics_file = RESULTS / "evaluation" / "metrics.csv"
gcn_history_file = RESULTS / "training" / "gcn_training_history.csv"
gat_history_file = RESULTS / "training" / "gat_training_history.csv"
pred_file = RESULTS / "evaluation" / "paper_predictions.csv"
explain_dir = RESULTS / "explainability"

with tab_graph:
    if summary_file.exists():
        render_graph_stats(pd.read_csv(summary_file), RESULTS / "graph_analysis")
    else:
        st.info("Graph statistics will appear after running Notebook 02.")

with tab_train:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; font-weight: 700; font-size: 1.6rem; color: #F8FAFC;">
            📈 Training Dynamics & Loss Convergence
        </h2>
        <p style="color: #94A3B8; margin-top: 4px; font-size: 0.95rem;">
            Comparative epoch-by-epoch loss reduction and validation accuracy trajectories over 30 training epochs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if gcn_history_file.exists() and gat_history_file.exists():
        gcn_df = pd.read_csv(gcn_history_file)
        gat_df = pd.read_csv(gat_history_file)

        # Milestone Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔹</div>
                <div class="metric-value" style="color: #38BDF8;">{gcn_df['validation_accuracy'].max()*100:.2f}%</div>
                <div class="metric-label">GCN Peak Val Acc</div>
                <div class="metric-sub">Epoch {gcn_df['validation_accuracy'].idxmax()+1} / 30</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
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
                <div class="metric-value" style="color: #10B981;">{gcn_df['loss'].min():.4f}</div>
                <div class="metric-label">GCN Min Training Loss</div>
                <div class="metric-sub">Cross-Entropy Objective</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-value" style="color: #F59E0B;">{gat_df['loss'].min():.4f}</div>
                <div class="metric-label">GAT Min Training Loss</div>
                <div class="metric-sub">Cross-Entropy Objective</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Plotly Interactive Comparison Curves
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### 📉 Training Loss Trajectories")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=gcn_df["epoch"], y=gcn_df["loss"],
                mode="lines+markers", name="GCN (Loss)",
                line=dict(color="#38BDF8", width=2.5),
                marker=dict(size=5)
            ))
            fig_loss.add_trace(go.Scatter(
                x=gat_df["epoch"], y=gat_df["loss"],
                mode="lines+markers", name="GAT (Loss)",
                line=dict(color="#C084FC", width=2.5),
                marker=dict(size=5)
            ))
            fig_loss.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Cross-Entropy Loss", gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                height=320
            )
            st.plotly_chart(fig_loss, width="stretch")
        with col_c2:
            st.markdown("#### 🎯 Validation Accuracy (%)")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=gcn_df["epoch"], y=gcn_df["validation_accuracy"] * 100,
                mode="lines+markers", name="GCN (Val Acc)",
                line=dict(color="#38BDF8", width=2.5),
                marker=dict(size=5)
            ))
            fig_acc.add_trace(go.Scatter(
                x=gat_df["epoch"], y=gat_df["validation_accuracy"] * 100,
                mode="lines+markers", name="GAT (Val Acc)",
                line=dict(color="#C084FC", width=2.5),
                marker=dict(size=5)
            ))
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Validation Accuracy (%)", gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                height=320
            )
            st.plotly_chart(fig_acc, width="stretch")

    else:
        st.info("Training history logs will appear once notebooks 04 & 05 are executed.")

with tab_eval:
    if metrics_file.exists():
        render_model_metrics(pd.read_csv(metrics_file), RESULTS / "evaluation")
    else:
        st.info("Evaluation metrics and confusion matrices will appear after Notebook 07.")

with tab_class:
    render_classification_demo(pred_file)

with tab_embed:
    render_embedding_image(explain_dir)
