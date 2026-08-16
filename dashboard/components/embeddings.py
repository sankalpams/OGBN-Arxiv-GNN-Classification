from pathlib import Path
import streamlit as st


def render_embedding_image(explain_dir: Path | None = None) -> None:
    if explain_dir is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        explain_dir = dashboard_dir.parent / "results" / "explainability"
    elif not isinstance(explain_dir, Path):
        explain_dir = Path(explain_dir)
        if explain_dir.is_file():
            explain_dir = explain_dir.parent

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; font-weight: 700; font-size: 1.6rem; color: #F8FAFC;">
            🌌 Latent Embeddings & Representation Explainability
        </h2>
        <p style="color: #94A3B8; margin-top: 4px; font-size: 0.95rem;">
            Dimensionality reduction and neighborhood homophily analysis of 256-dimensional learned GCN representations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Embedding Key Findings
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value">256-dim</div>
            <div class="metric-label">Hidden Representation</div>
            <div class="metric-sub">Layer 1 GCN Latent Space</div>
        </div>
        """, unsafe_allow_html=True)
    with e2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🌐</div>
            <div class="metric-value" style="color: #10B981;">85.71%</div>
            <div class="metric-label">1-Hop Homophily Agreement</div>
            <div class="metric-sub">Strong Intra-Field Clustering</div>
        </div>
        """, unsafe_allow_html=True)
    with e3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🔭</div>
            <div class="metric-value" style="color: #38BDF8;">40 Clusters</div>
            <div class="metric-label">Manifold Separation</div>
            <div class="metric-sub">Clear Subfield Partitions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    pca_plot = explain_dir / "pca_embeddings.png"
    tsne_plot = explain_dir / "tsne_embeddings.png"

    tab_tsne, tab_pca, tab_side = st.tabs([
        "🌀 Non-Linear t-SNE Manifold (Recommended)",
        "📊 Linear PCA 2D Projection",
        "⚖️ Side-by-Side Dual View"
    ])

    with tab_tsne:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); padding: 14px 18px; border-radius: 10px; border-left: 4px solid #C084FC; margin-bottom: 15px;">
            <span style="font-weight: 600; color: #C084FC;">🌀 t-SNE Non-Linear Manifold:</span>
            <span style="color: #CBD5E1; font-size: 0.9rem;">
                Resolves high-density non-linear semantic clusters corresponding to specialized communities such as Computer Vision (<code>cs.CV</code>), Machine Learning (<code>cs.LG</code>), Artificial Intelligence (<code>cs.AI</code>), and Cryptography (<code>cs.CR</code>).
            </span>
        </div>
        """, unsafe_allow_html=True)
        if tsne_plot.exists():
            st.image(str(tsne_plot), caption="t-SNE 2D Manifold Embedding Clustered by Research Topic", width="stretch")
        else:
            st.info("Run Notebook 08 to generate t-SNE embedding visualization.")

    with tab_pca:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); padding: 14px 18px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-bottom: 15px;">
            <span style="font-weight: 600; color: #38BDF8;">📊 PCA Linear Projection:</span>
            <span style="color: #CBD5E1; font-size: 0.9rem;">
                Demonstrates macroscopic variance separation along principal orthogonal axes between broad academic disciplines (e.g., Theoretical Computer Science vs Applied Deep Learning).
            </span>
        </div>
        """, unsafe_allow_html=True)
        if pca_plot.exists():
            st.image(str(pca_plot), caption="PCA 2D Linear Projection of Learned Node Embeddings (Colored by Subject Area)", width="stretch")
        else:
            st.info("Run Notebook 08 to generate PCA embedding visualization.")

    with tab_side:
        c1, c2 = st.columns(2)
        with c1:
            if pca_plot.exists():
                st.image(str(pca_plot), caption="PCA (Linear Global Geometry)", width="stretch")
        with c2:
            if tsne_plot.exists():
                st.image(str(tsne_plot), caption="t-SNE (Non-Linear Local Manifolds)", width="stretch")
