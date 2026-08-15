from pathlib import Path
import streamlit as st


def render_model_metrics(metrics_df, eval_dir: Path | None = None) -> None:
    st.subheader("Model Performance Comparison (Test Split)")
    
    if eval_dir is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        eval_dir = dashboard_dir.parent / "results" / "evaluation"
        
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    comp_plot = eval_dir / "model_comparison.png"
    if comp_plot.exists():
        st.image(str(comp_plot), caption="Accuracy, Precision, Recall & F1-Score Comparison", use_container_width=True)
        
    st.markdown("---")
    st.write("**Confusion Matrices:**")
    c1, c2 = st.columns(2)
    
    cm_gcn = eval_dir / "confusion_matrix_gcn.png"
    cm_gat = eval_dir / "confusion_matrix_gat.png"
    
    with c1:
        if cm_gcn.exists():
            st.image(str(cm_gcn), caption="GCN Confusion Matrix (40 Classes)", use_container_width=True)
    with c2:
        if cm_gat.exists():
            st.image(str(cm_gat), caption="GAT Confusion Matrix (40 Classes)", use_container_width=True)

