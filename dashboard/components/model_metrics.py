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
        project_root / "results" / "evaluation" / candidate_name,
        project_root / "results" / candidate_name,
        dashboard_dir / "results" / "evaluation" / candidate_name,
        dashboard_dir / "results" / candidate_name,
        Path.cwd() / "results" / "evaluation" / candidate_name,
        Path.cwd() / "results" / candidate_name,
        Path("results/evaluation") / candidate_name,
        Path("results") / candidate_name,
    ])
    
    for path in candidates:
        if path.exists():
            return path
    return None


def render_model_metrics(metrics_df: pd.DataFrame, eval_dir: Path | None = None) -> None:
    if eval_dir is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        eval_dir = dashboard_dir.parent / "results" / "evaluation"

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; font-weight: 700; font-size: 1.6rem; color: #F8FAFC;">
            🏆 Model Benchmark & Performance Comparison
        </h2>
        <p style="color: #94A3B8; margin-top: 4px; font-size: 0.95rem;">
            Comparative empirical evaluation on the 48,603 held-out test papers (2019–2020 chronological partition).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Highlight Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card highlight-gcn">
            <div class="metric-icon">🥇</div>
            <div class="metric-value" style="color: #38BDF8;">58.64%</div>
            <div class="metric-label">GCN Test Accuracy</div>
            <div class="metric-sub" style="color: #10B981;">▲ +1.25% vs GAT</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🥈</div>
            <div class="metric-value" style="color: #C084FC;">57.39%</div>
            <div class="metric-label">GAT Test Accuracy</div>
            <div class="metric-sub">Multi-Head Attention (4 heads)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value" style="color: #10B981;">3.25x</div>
            <div class="metric-label">Speed Advantage (GCN)</div>
            <div class="metric-sub">2.4s/epoch vs 7.8s/epoch</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value" style="color: #F59E0B;">53.89%</div>
            <div class="metric-label">GCN Weighted F1</div>
            <div class="metric-sub">GAT: 52.20% (▲ +1.69%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Interactive Comparison Chart & Metric Breakdown
    col_chart, col_radar = st.columns([1.6, 1.4])

    with col_chart:
        st.markdown("#### 📊 Metric-by-Metric Empirical Comparison")
        
        # Prepare melted DataFrame for Plotly
        plot_df = pd.DataFrame([
            {"Model": "GCN", "Metric": "Test Accuracy", "Score (%)": 58.64},
            {"Model": "GAT", "Metric": "Test Accuracy", "Score (%)": 57.39},
            {"Model": "GCN", "Metric": "Weighted Precision", "Score (%)": 54.92},
            {"Model": "GAT", "Metric": "Weighted Precision", "Score (%)": 54.14},
            {"Model": "GCN", "Metric": "Weighted Recall", "Score (%)": 58.64},
            {"Model": "GAT", "Metric": "Weighted Recall", "Score (%)": 57.39},
            {"Model": "GCN", "Metric": "Weighted F1-Score", "Score (%)": 53.89},
            {"Model": "GAT", "Metric": "Weighted F1-Score", "Score (%)": 52.20},
        ])

        fig_bar = px.bar(
            plot_df,
            x="Metric",
            y="Score (%)",
            color="Model",
            barmode="group",
            text="Score (%)",
            color_discrete_map={"GCN": "#38BDF8", "GAT": "#A855F7"}
        )
        fig_bar.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            marker=dict(line=dict(width=1, color="#1E293B"))
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            yaxis=dict(range=[45, 65], title="Score (%)", gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_radar:
        st.markdown("#### 🕸️ Multidimensional Performance Radar")
        categories = ["Accuracy", "Precision", "Recall", "F1-Score", "Epoch Efficiency"]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[58.64, 54.92, 58.64, 53.89, 90.0],
            theta=categories,
            fill='toself',
            name='GCN (Spectral)',
            line=dict(color='#38BDF8', width=2),
            fillcolor='rgba(56, 189, 248, 0.2)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[57.39, 54.14, 57.39, 52.20, 45.0],
            theta=categories,
            fill='toself',
            name='GAT (Attention)',
            line=dict(color='#A855F7', width=2),
            fillcolor='rgba(168, 85, 247, 0.2)'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[30, 100], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(l=30, r=30, t=30, b=20),
            height=320
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Detailed Table
    st.markdown("#### 📋 Raw Metric Values")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # High-Res Confusion Matrices
    st.markdown("#### 🎯 40-Class Test Confusion Matrices")
    tab_cm_gcn, tab_cm_gat, tab_cm_comp = st.tabs(["🔹 GCN Matrix", "🔸 GAT Matrix", "📊 Side-by-Side Comparison"])

    cm_gcn = _resolve_candidate_file("confusion_matrix_gcn.png", eval_dir)
    cm_gat = _resolve_candidate_file("confusion_matrix_gat.png", eval_dir)
    comp_plot = _resolve_candidate_file("model_comparison.png", eval_dir)

    with tab_cm_gcn:
        if cm_gcn and cm_gcn.exists():
            st.image(str(cm_gcn), caption="GCN Confusion Matrix across 40 arXiv Subject Categories", use_container_width=True)
    with tab_cm_gat:
        if cm_gat and cm_gat.exists():
            st.image(str(cm_gat), caption="GAT Confusion Matrix across 40 arXiv Subject Categories", use_container_width=True)
    with tab_cm_comp:
        if comp_plot and comp_plot.exists():
            st.image(str(comp_plot), caption="Full Comparative Bar Plot across All Metrics", use_container_width=True)
