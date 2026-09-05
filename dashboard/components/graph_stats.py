import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _resolve_candidate_file(candidate_name: str, preferred_dir: Path | None = None) -> Path | None:
    """Multi-path search to reliably resolve result files across local and cloud environments."""
    candidates = []
    if preferred_dir is not None:
        candidates.append(preferred_dir / candidate_name)
        candidates.append(preferred_dir.parent / candidate_name)
    
    current_dir = Path(__file__).resolve().parent
    dashboard_dir = current_dir.parent
    project_root = dashboard_dir.parent
    
    candidates.extend([
        project_root / "results" / "graph_analysis" / candidate_name,
        project_root / "results" / candidate_name,
        dashboard_dir / "results" / "graph_analysis" / candidate_name,
        dashboard_dir / "results" / candidate_name,
        Path.cwd() / "results" / "graph_analysis" / candidate_name,
        Path.cwd() / "results" / candidate_name,
        Path("results/graph_analysis") / candidate_name,
        Path("results") / candidate_name,
    ])
    
    for path in candidates:
        if path.exists():
            return path
    return None


def render_3d_citation_graph(json_path: Path | None = None) -> go.Figure | None:
    target_path = None
    if json_path is not None and json_path.exists():
        target_path = json_path
    else:
        target_path = _resolve_candidate_file("subgraph_3d.json", json_path.parent if json_path else None)
        
    if target_path is None or not target_path.exists():
        return None
    
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        return None
        
    node_dict = {n["id"]: n for n in nodes}
    
    # 3D Edges Trace
    edge_x, edge_y, edge_z = [], [], []
    for edge in edges:
        u_id = edge.get("source")
        v_id = edge.get("target")
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

    # Header section with Liquid Glass Typography
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="margin: 0; font-weight: 800; font-size: 1.7rem; color: #FFFFFF; letter-spacing: -0.02em;">
            📊 OGBN-Arxiv Network Topology & Macro Properties
        </h2>
        <p style="color: #94A3B8; margin-top: 5px; font-size: 0.98rem; line-height: 1.5;">
            Topological characterization of 169,343 computer science publications and 2.3M directed citation links with liquid graph manifolds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # High-impact Liquid Glass KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card highlight-gcn">
            <div class="metric-icon">📄</div>
            <div class="metric-value" style="color: #38BDF8;">169,343</div>
            <div class="metric-label">Total Nodes (Papers)</div>
            <div class="metric-sub">Microsoft Academic Graph</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card highlight-gat">
            <div class="metric-icon">🔗</div>
            <div class="metric-value" style="color: #C084FC;">2,315,598</div>
            <div class="metric-label">Directed Edges</div>
            <div class="metric-sub">Avg Degree: 13.67 links/node</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧬</div>
            <div class="metric-value" style="color: #34D399;">128-dim</div>
            <div class="metric-label">Feature Embeddings</div>
            <div class="metric-sub">Word2Vec Title+Abstract</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value" style="color: #FB923C;">40 Classes</div>
            <div class="metric-label">arXiv CS Categories</div>
            <div class="metric-sub">Single-label Subject Class</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Interactive Partitioning Donut Chart & Topology Summary in Liquid Glass
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown("""
        <div class="glass-img-container">
            <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 8px;">🕒 Chronological Temporal Splits</div>
        """, unsafe_allow_html=True)
        split_data = pd.DataFrame({
            "Split": ["Train (≤ 2017)", "Validation (2018)", "Test (2019-2020)"],
            "Papers": [90941, 29799, 48603],
            "Percentage": [53.7, 17.6, 28.7]
        })
        fig_donut = px.pie(
            split_data,
            values="Papers",
            names="Split",
            hole=0.58,
            color="Split",
            color_discrete_map={
                "Train (≤ 2017)": "#38BDF8",
                "Validation (2018)": "#FB923C",
                "Test (2019-2020)": "#34D399"
            }
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent+label",
            marker=dict(line=dict(color="rgba(11, 17, 32, 0.8)", width=2))
        )
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(l=15, r=15, t=15, b=15),
            height=270,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0")
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-img-container">
            <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 8px;">📐 Topological Metrics Summary</div>
        """, unsafe_allow_html=True)
        # Format stats nicely
        formatted_df = stats_df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype == float:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.6f}" if abs(x) < 0.01 else f"{x:.2f}")
            elif formatted_df[col].dtype == int or "int" in str(formatted_df[col].dtype):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,}")
        
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%); padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3); border-top: 1px solid rgba(255, 255, 255, 0.25); margin-top: 10px; box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.15);">
            <div style="font-weight: 700; color: #38BDF8; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
                <span>💡 Homophily Insight</span>
            </div>
            <div style="color: #CBD5E1; font-size: 0.84rem; margin-top: 3px; line-height: 1.45;">
                The ogbn-arxiv network exhibits strong domain homophily (85.7% local neighbor agreement), which enables isotropic Laplacian smoothing in GCNs to outperform complex attention kernels.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

    # 3D Interactive Topology & Visual Artifacts
    st.markdown("#### 🔬 Network Visualizations & 3D Interactive Topology")
    tab_3d_graph, tab_2d_plots = st.tabs([
        "🕸️ 3D Citation Subgraph (Scroll & Orbit)",
        "📊 2D Distribution & Topology Artifacts"
    ])

    subgraph_3d_file = _resolve_candidate_file("subgraph_3d.json", graph_analysis_dir)
    deg_plot = _resolve_candidate_file("degree_distribution.png", graph_analysis_dir)
    subgraph_plot = _resolve_candidate_file("sample_subgraph.png", graph_analysis_dir)

    with tab_3d_graph:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%); padding: 14px 18px; border-radius: 14px; border: 1px solid rgba(56, 189, 248, 0.3); border-top: 1px solid rgba(255, 255, 255, 0.25); margin-bottom: 14px; box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.4);">
            <span style="font-weight: 800; color: #38BDF8;">🕹️ 3D Graph Scroll Controls:</span>
            <span style="color: #CBD5E1; font-size: 0.88rem;">
                Use <b>Mouse Scroll Wheel</b> to zoom in/out in 3D space, <b>Click & Drag</b> to rotate 360°, and <b>Hover</b> on node spheres to inspect local paper citation degrees.
            </span>
        </div>
        """, unsafe_allow_html=True)
        fig_3d = render_3d_citation_graph(subgraph_3d_file)
        if fig_3d is not None:
            st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("3D Subgraph data file not found.")

    with tab_2d_plots:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if deg_plot and deg_plot.exists():
                st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
                st.image(str(deg_plot), caption="📊 Power-Law Degree Distribution (Log Scale)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
        with col_img2:
            if subgraph_plot and subgraph_plot.exists():
                st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
                st.image(str(subgraph_plot), caption="🕸️ Local 2-Hop Ego Citation Subgraph Structure", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
