from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_graph_stats(stats_df: pd.DataFrame, graph_analysis_dir: Path | None = None) -> None:
    if graph_analysis_dir is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        graph_analysis_dir = dashboard_dir.parent / "results" / "graph_analysis"

    # Header section
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; font-weight: 700; font-size: 1.6rem; color: #F8FAFC;">
            📊 OGBN-Arxiv Network Topology & Macro Properties
        </h2>
        <p style="color: #94A3B8; margin-top: 4px; font-size: 0.95rem;">
            Topological characterization of 169,343 computer science publications and 2.3M directed citation links.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # High-impact KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📄</div>
            <div class="metric-value">169,343</div>
            <div class="metric-label">Total Nodes (Papers)</div>
            <div class="metric-sub">Microsoft Academic Graph</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🔗</div>
            <div class="metric-value">2,315,598</div>
            <div class="metric-label">Directed Edges</div>
            <div class="metric-sub">Avg Degree: 13.67 links/node</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧬</div>
            <div class="metric-value">128-dim</div>
            <div class="metric-label">Feature Embeddings</div>
            <div class="metric-sub">Word2Vec Title+Abstract</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">40 Classes</div>
            <div class="metric-label">arXiv CS Categories</div>
            <div class="metric-sub">Single-label Subject Class</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Interactive Partitioning Donut Chart & Topology Summary
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown("#### 🕒 Chronological Temporal Splits")
        split_data = pd.DataFrame({
            "Split": ["Train (≤ 2017)", "Validation (2018)", "Test (2019-2020)"],
            "Papers": [90941, 29799, 48603],
            "Percentage": [53.7, 17.6, 28.7]
        })
        fig_donut = px.pie(
            split_data,
            values="Papers",
            names="Split",
            hole=0.55,
            color="Split",
            color_discrete_map={
                "Train (≤ 2017)": "#38BDF8",
                "Validation (2018)": "#F59E0B",
                "Test (2019-2020)": "#10B981"
            }
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#0B0F19", width=2))
        )
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0")
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("#### 📐 Topological Metrics Summary")
        # Format stats nicely
        formatted_df = stats_df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype == float:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.6f}" if abs(x) < 0.01 else f"{x:.2f}")
            elif formatted_df[col].dtype == int or "int" in str(formatted_df[col].dtype):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,}")
        
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); padding: 14px 18px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-top: 10px;">
            <span style="font-weight: 600; color: #38BDF8;">💡 Homophily Insight:</span>
            <span style="color: #CBD5E1; font-size: 0.9rem;">
                The ogbn-arxiv citation network exhibits strong domain homophily (85.7% local neighbor agreement), which enables isotropic Laplacian smoothing in GCNs to outperform complex attention kernels.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Visual Artifacts
    st.markdown("#### 🔬 Empirical Topological Visualizations")
    col_img1, col_img2 = st.columns(2)
    
    deg_plot = graph_analysis_dir / "degree_distribution.png"
    subgraph_plot = graph_analysis_dir / "sample_subgraph.png"
    
    with col_img1:
        if deg_plot.exists():
            st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
            st.image(str(deg_plot), caption="📊 Power-Law Degree Distribution (Log Scale)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
    with col_img2:
        if subgraph_plot.exists():
            st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
            st.image(str(subgraph_plot), caption="🕸️ Local 2-Hop Ego Citation Subgraph Structure", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
