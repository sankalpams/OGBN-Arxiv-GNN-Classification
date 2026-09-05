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
        project_root / "results" / "explainability" / candidate_name,
        project_root / "results" / candidate_name,
        dashboard_dir / "results" / "explainability" / candidate_name,
        dashboard_dir / "results" / candidate_name,
        Path.cwd() / "results" / "explainability" / candidate_name,
        Path.cwd() / "results" / candidate_name,
        Path("results/explainability") / candidate_name,
        Path("results") / candidate_name,
    ])
    
    for path in candidates:
        if path.exists():
            return path
    return None


def render_embedding_image(explain_dir: Path | None = None) -> None:
    if explain_dir is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        explain_dir = dashboard_dir.parent / "results" / "explainability"
    elif not isinstance(explain_dir, Path):
        explain_dir = Path(explain_dir)
        if explain_dir.is_file():
            explain_dir = explain_dir.parent

    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="margin: 0; font-weight: 800; font-size: 1.7rem; color: #FFFFFF; letter-spacing: -0.02em;">
            🌌 Latent Embeddings & 3D Interactive Manifold Explorer
        </h2>
        <p style="color: #94A3B8; margin-top: 5px; font-size: 0.98rem; line-height: 1.5;">
            Dimensionality reduction and neighborhood homophily analysis of 256-dimensional learned GCN representations with full <b>3D scroll-to-zoom</b> and orbital rotation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Liquid Glass Embedding Key Findings
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value" style="color: #C084FC;">256-dim</div>
            <div class="metric-label">Hidden Representation</div>
            <div class="metric-sub">Layer 1 GCN Latent Space</div>
        </div>
        """, unsafe_allow_html=True)
    with e2:
        st.markdown("""
        <div class="metric-card highlight-gcn">
            <div class="metric-icon">🌐</div>
            <div class="metric-value" style="color: #34D399;">85.71%</div>
            <div class="metric-label">1-Hop Homophily Agreement</div>
            <div class="metric-sub">Strong Intra-Field Clustering</div>
        </div>
        """, unsafe_allow_html=True)
    with e3:
        st.markdown("""
        <div class="metric-card highlight-gat">
            <div class="metric-icon">🔭</div>
            <div class="metric-value" style="color: #38BDF8;">40 Clusters</div>
            <div class="metric-label">Manifold Separation</div>
            <div class="metric-sub">Clear Subfield Partitions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    pca_plot = _resolve_candidate_file("pca_embeddings.png", explain_dir)
    tsne_plot = _resolve_candidate_file("tsne_embeddings.png", explain_dir)
    emb_3d_file = _resolve_candidate_file("embeddings_3d.csv", explain_dir)

    tab_3d, tab_tsne, tab_pca, tab_side = st.tabs([
        "🎮 3D Interactive Scroll & Orbit Explorer",
        "🌀 Non-Linear t-SNE 2D Manifold",
        "📊 Linear PCA 2D Projection",
        "⚖️ Side-by-Side Dual View"
    ])

    with tab_3d:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%); padding: 14px 18px; border-radius: 14px; border: 1px solid rgba(56, 189, 248, 0.3); border-top: 1px solid rgba(255, 255, 255, 0.25); margin-bottom: 15px; box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.4);">
            <div style="font-weight: 800; color: #38BDF8; font-size: 1.05rem;">🕹️ 3D Scroll & Orbit Navigation Controls</div>
            <div style="color: #CBD5E1; font-size: 0.88rem; margin-top: 4px; line-height: 1.5;">
                • <b>Scroll Mouse Wheel</b>: Smooth 3D Zoom In / Out &nbsp;|&nbsp;
                • <b>Left Click + Drag</b>: Free 360° 3D Orbital Rotation &nbsp;|&nbsp;
                • <b>Right Click + Drag</b>: 3D Camera Pan &nbsp;|&nbsp;
                • <b>Hover Node</b>: Inspect MAG Paper ID, True Subject, and Model Predictions
            </div>
        </div>
        """, unsafe_allow_html=True)

    if emb_3d_file and emb_3d_file.exists():
        with tab_3d:
            df_3d = pd.read_csv(emb_3d_file)

            st.markdown("""
            <div class="glass-img-container" style="padding: 16px 20px; margin-bottom: 14px;">
            """, unsafe_allow_html=True)
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 1.2])
            with col_ctrl1:
                proj_mode = st.radio(
                    "3D Manifold Technique:",
                    ["t-SNE (Non-Linear Semantic Clusters)", "PCA (Linear Principal Axes)"],
                    horizontal=True
                )
            with col_ctrl2:
                categories = ["All 40 Categories"] + sorted(df_3d["true_label"].dropna().unique().tolist())
                chosen_cat = st.selectbox("🎯 Filter 3D Point Cloud by Category:", categories)
            with col_ctrl3:
                pt_size = st.slider("Node Point Size:", min_value=2, max_value=10, value=4)
            st.markdown("</div>", unsafe_allow_html=True)

            filtered_3d = df_3d.copy()
            if chosen_cat != "All 40 Categories":
                filtered_3d = filtered_3d[filtered_3d["true_label"] == chosen_cat]

            x_col = "tsne_x" if "t-SNE" in proj_mode else "pca_x"
            y_col = "tsne_y" if "t-SNE" in proj_mode else "pca_y"
            z_col = "tsne_z" if "t-SNE" in proj_mode else "pca_z"

            fig_3d = px.scatter_3d(
                filtered_3d,
                x=x_col,
                y=y_col,
                z=z_col,
                color="true_label",
                hover_name="true_label",
                hover_data={
                    "node_id": True,
                    "paper_id": True,
                    "gcn_pred": True,
                    "gat_pred": True,
                    x_col: False,
                    y_col: False,
                    z_col: False,
                    "true_label": False
                },
                labels={
                    x_col: "Dimension 1",
                    y_col: "Dimension 2",
                    z_col: "Dimension 3",
                    "true_label": "Category"
                },
                opacity=0.88
            )

            fig_3d.update_traces(
                marker=dict(size=pt_size, line=dict(width=0))
            )

            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(11, 17, 32, 0.8)", gridcolor="rgba(255, 255, 255, 0.1)", showbackground=True, title="Latent Dim 1"),
                    yaxis=dict(backgroundcolor="rgba(11, 17, 32, 0.8)", gridcolor="rgba(255, 255, 255, 0.1)", showbackground=True, title="Latent Dim 2"),
                    zaxis=dict(backgroundcolor="rgba(11, 17, 32, 0.8)", gridcolor="rgba(255, 255, 255, 0.1)", showbackground=True, title="Latent Dim 3"),
                    camera=dict(
                        eye=dict(x=1.6, y=1.6, z=1.3)
                    )
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                margin=dict(l=0, r=0, t=0, b=0),
                height=560,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.95,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=9)
                )
            )

            st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        with tab_3d:
            st.info("3D Latent coordinates file (embeddings_3d.csv) not found.")

    with tab_tsne:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(192, 132, 252, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%); padding: 14px 18px; border-radius: 14px; border: 1px solid rgba(192, 132, 252, 0.3); border-top: 1px solid rgba(255, 255, 255, 0.25); margin-bottom: 15px; box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.4);">
            <span style="font-weight: 700; color: #C084FC;">🌀 t-SNE Non-Linear Manifold:</span>
            <span style="color: #CBD5E1; font-size: 0.9rem;">
                Resolves high-density non-linear semantic clusters corresponding to specialized communities such as Computer Vision (<code>cs.CV</code>), Machine Learning (<code>cs.LG</code>), Artificial Intelligence (<code>cs.AI</code>), and Cryptography (<code>cs.CR</code>).
            </span>
        </div>
        """, unsafe_allow_html=True)
        if tsne_plot and tsne_plot.exists():
            st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
            st.image(str(tsne_plot), caption="t-SNE 2D Manifold Embedding Clustered by Research Topic", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Run Notebook 08 to generate t-SNE embedding visualization.")

    with tab_pca:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%); padding: 14px 18px; border-radius: 14px; border: 1px solid rgba(56, 189, 248, 0.3); border-top: 1px solid rgba(255, 255, 255, 0.25); margin-bottom: 15px; box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.4);">
            <span style="font-weight: 700; color: #38BDF8;">📊 PCA Linear Projection:</span>
            <span style="color: #CBD5E1; font-size: 0.9rem;">
                Demonstrates macroscopic variance separation along principal orthogonal axes between broad academic disciplines (e.g., Theoretical Computer Science vs Applied Deep Learning).
            </span>
        </div>
        """, unsafe_allow_html=True)
        if pca_plot and pca_plot.exists():
            st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
            st.image(str(pca_plot), caption="PCA 2D Linear Projection of Learned Node Embeddings (Colored by Subject Area)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Run Notebook 08 to generate PCA embedding visualization.")

    with tab_side:
        c1, c2 = st.columns(2)
        with c1:
            if pca_plot and pca_plot.exists():
                st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
                st.image(str(pca_plot), caption="PCA (Linear Global Geometry)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            if tsne_plot and tsne_plot.exists():
                st.markdown("<div class='glass-img-container'>", unsafe_allow_html=True)
                st.image(str(tsne_plot), caption="t-SNE (Non-Linear Local Manifolds)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
