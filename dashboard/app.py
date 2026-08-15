from pathlib import Path
import sys

# Ensure dashboard directory is on path for component imports
_dashboard_dir = Path(__file__).resolve().parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

# Ensure project root is on path for src imports
_project_root = _dashboard_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

ROOT = _project_root
RESULTS = ROOT / "results"

from components.graph_stats import render_graph_stats
from components.model_metrics import render_model_metrics
from components.classification import render_classification_demo
from components.embeddings import render_embedding_image

st.set_page_config(
    page_title="CCS4354 - OGBN-Arxiv Graph Intelligence Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header
st.title("🕸️ CCS4354 - OGBN-Arxiv Graph Intelligence Dashboard")
st.caption("Deep Learning on Graphs: GCN vs. GAT Node Classification on the 169K-Node OGB Citation Network")

# Sidebar summary
with st.sidebar:
    st.header("📌 Project Status")
    st.success("✅ All 8 Notebooks Executed")
    st.markdown("""
    **Dataset:** `ogbn-arxiv`
    - **Nodes:** 169,343
    - **Edges:** 2,315,598
    - **Classes:** 40
    - **Feature Dim:** 128
    
    **Architectures:**
    - **GCN:** 2-Layer (256 hidden)
    - **GAT:** 2-Layer (64 hidden, 4 heads)
    """)
    st.markdown("---")
    st.caption("CCS4354 Tensors and Graphs Assignment")

# Main Navigation Tabs
tab_graph, tab_train, tab_eval, tab_class, tab_embed = st.tabs([
    "📊 Graph Analysis",
    "📈 Training Dynamics",
    "🏆 Model Evaluation",
    "🔬 Node Classification Lookup",
    "🌌 Embeddings & Explainability"
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
    st.subheader("GNN Training & Validation Convergence")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### 🔹 GCN Training History")
        if gcn_history_file.exists():
            gcn_df = pd.read_csv(gcn_history_file)
            st.line_chart(gcn_df.set_index("epoch")[["loss"]], use_container_width=True)
            st.line_chart(gcn_df.set_index("epoch")[["train_accuracy", "validation_accuracy"]], use_container_width=True)
        else:
            st.info("GCN training history will appear after Notebook 04.")
            
    with col_t2:
        st.markdown("#### 🔸 GAT Training History")
        if gat_history_file.exists():
            gat_df = pd.read_csv(gat_history_file)
            st.line_chart(gat_df.set_index("epoch")[["loss"]], use_container_width=True)
            st.line_chart(gat_df.set_index("epoch")[["train_accuracy", "validation_accuracy"]], use_container_width=True)
        else:
            st.info("GAT training history will appear after Notebook 05.")

with tab_eval:
    if metrics_file.exists():
        render_model_metrics(pd.read_csv(metrics_file), RESULTS / "evaluation")
    else:
        st.info("Evaluation metrics and confusion matrices will appear after Notebook 07.")

with tab_class:
    render_classification_demo(pred_file)

with tab_embed:
    render_embedding_image(explain_dir)

