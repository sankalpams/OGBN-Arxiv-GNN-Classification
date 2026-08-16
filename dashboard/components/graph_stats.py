import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_3d_citation_graph(json_path: Path) -> go.Figure | None:
    if not json_path.exists():
        return None
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        return None
        
    node_dict = {n["id"]: n for n in nodes}
    
    # 3D Edges Trace
    edge_x, edge_y, edge_z = [], [], []
    for edge in edges:
        u_id = edge["source"]
        v_id = edge["target"]
        if u_id in node_dict and v_id in node_dict:
            u = node_dict[u_id]
            v = node_dict[v_id]
            edge_x.extend([u["x"], v["x"], None])
            edge_y.extend([u["y"], v["y"], None])
            edge_z.extend([u["z"], v["z"], None])
            
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(56, 189, 248, 0.35)", width=2),
        hoverinfo="none",
        name="Citation Links"
    )
    
    # 3D Nodes Trace
    df_nodes = pd.DataFrame(nodes)
    
    fig = go.Figure(data=[edge_trace])
    
    # Add nodes grouped by category for clean legend
    categories = df_nodes["category"].unique()
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    
    for i, cat in enumerate(categories):
        subset = df_nodes[df_nodes["category"] == cat]
        fig.add_trace(go.Scatter3d(
            x=subset["x"],
            y=subset["y"],
            z=subset["z"],
            mode="markers",
            name=str(cat),
            marker=dict(
                size=subset["degree"].apply(lambda d: max(5, min(14, 5 + d * 1.2))),
                color=colors[i % len(colors)],
                line=dict(width=1, color="#0F172A"),
                opacity=0.9
            ),
            customdata=subset[["id", "degree", "gcn_pred"]],
            hovertemplate=(
                "<b>Node #%{customdata[0]}</b><br>"
                "Category: " + str(cat) + "<br>"
                "Local Degree: %{customdata[1]} links<br>"
                "GCN Pred: %{customdata[2]}<extra></extra>"
            )
        ))
        
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=9)
        )
    )
    return fig


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
        st.plotly_chart(fig_donut, width="stretch")

    with col_right:
        st.markdown("#### 📐 Topological Metrics Summary")
        # Format stats nicely
        formatted_df = stats_df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype == float:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.6f}" if abs(x) < 0.01 else f"{x:.2f}")
            elif formatted_df[col].dtype == int or "int" in str(formatted_df[col].dtype):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,}")
        
        st.dataframe(formatted_df, width="stretch", hide_index=True)
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); padding: 14px 18px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-top: 10px;">
            <span style="font-weight: 600; color: #38BDF8;">💡 Homophily Insight:</span>
            <span style="color: #CBD5E1; font-size: 0.9rem;">
                The ogbn-arxiv citation network exhibits strong domain homophily (85.7% local neighbor agreement), which enables isotropic Laplacian smoothing in GCNs to outperform complex attention kernels.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3D Interactive Topology & Visual Artifacts
    st.markdown("#### 🔬 Network Visualizations & 3D Interactive Topology")
    tab_3d_graph, tab_2d_plots = st.tabs([
        "🕸️ 3D Citation Subgraph (Scroll & Orbit)",
        "📊 2D Distribution & Topology Artifacts"
    ])

    subgraph_3d_file = graph_analysis_dir / "subgraph_3d.json"
    deg_plot = graph_analysis_dir / "degree_distribution.png"
    subgraph_plot = graph_analysis_dir / "sample_subgraph.png"

    with tab_3d_graph:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px 16px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-bottom: 12px;">
            <span style="font-weight: 700; color: #38BDF8;">🕹️ 3D Graph Scroll Controls:</span>
            <span style="color: #CBD5E1; font-size: 0.88rem;">
                Use <b>Mouse Scroll Wheel</b> to zoom in/out in 3D space, <b>Click & Drag</b> to rotate 360°, and <b>Hover</b> on node spheres to inspect local paper citation degrees.
            </span>
        </div>
        """, unsafe_allow_html=True)
        fig_3d = render_3d_citation_graph(subgraph_3d_file)
        if fig_3d is not None:
            st.plotly_chart(fig_3d, width="stretch")
        else:
            st.info("3D Subgraph data file not found.")

    with tab_2d_plots:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if deg_plot.exists():
                st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
                st.image(str(deg_plot), caption="📊 Power-Law Degree Distribution (Log Scale)", width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
                
        with col_img2:
            if subgraph_plot.exists():
                st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
                st.image(str(subgraph_plot), caption="🕸️ Local 2-Hop Ego Citation Subgraph Structure", width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
