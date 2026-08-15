from pathlib import Path
import streamlit as st


def render_embedding_image(explain_dir: Path | None = None) -> None:
    st.subheader("Latent Embedding & Representation Analysis")
    
    if explain_dir is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        explain_dir = dashboard_dir.parent / "results" / "explainability"
    elif not isinstance(explain_dir, Path):
        explain_dir = Path(explain_dir)
        if explain_dir.is_file():
            explain_dir = explain_dir.parent
            
    pca_plot = explain_dir / "pca_embeddings.png"
    tsne_plot = explain_dir / "tsne_embeddings.png"
    
    tab_pca, tab_tsne = st.tabs(["📊 PCA 2D Projection", "🌀 t-SNE Non-Linear Manifold"])
    
    with tab_pca:
        if pca_plot.exists():
            st.image(str(pca_plot), caption="PCA 2D Projection of Learned Node Embeddings (Colored by Subject Area)", use_container_width=True)
        else:
            st.info("Run Notebook 08 to generate PCA embedding visualisation.")
            
    with tab_tsne:
        if tsne_plot.exists():
            st.image(str(tsne_plot), caption="t-SNE 2D Manifold Embedding Clustered by Research Topic", use_container_width=True)
        else:
            st.info("Run Notebook 08 to generate t-SNE embedding visualisation.")

