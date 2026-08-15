from pathlib import Path
import streamlit as st


def render_graph_stats(stats_df, graph_analysis_dir: Path | None = None) -> None:
    st.subheader("OGBN-Arxiv Graph Overview")
    
    if graph_analysis_dir is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        graph_analysis_dir = dashboard_dir.parent / "results" / "graph_analysis"

    # Display KPI metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Nodes (Papers)", "169,343")
    kpi2.metric("Edges (Citations)", "2,315,598")
    kpi3.metric("Node Features", "128-dim")
    kpi4.metric("Subject Classes", "40 Categories")
    
    st.markdown("---")
    st.write("**Graph Structural Summary:**")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    deg_plot = graph_analysis_dir / "degree_distribution.png"
    subgraph_plot = graph_analysis_dir / "sample_subgraph.png"
    
    with col1:
        if deg_plot.exists():
            st.image(str(deg_plot), caption="Node In-Degree / Out-Degree Distribution", use_container_width=True)
    with col2:
        if subgraph_plot.exists():
            st.image(str(subgraph_plot), caption="Sampled Local Subgraph Topology", use_container_width=True)

