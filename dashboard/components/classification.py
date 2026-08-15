from pathlib import Path
import pandas as pd
import streamlit as st


def render_classification_demo(predictions_file: Path | None = None) -> None:
    st.subheader("Node Classification & Paper Predictions")
    
    if predictions_file is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        predictions_file = dashboard_dir.parent / "results" / "evaluation" / "paper_predictions.csv"
        
    if not predictions_file.exists():
        st.info("Run the evaluation notebook to export paper-level predictions for interactive lookup.")
        return

    df = pd.read_csv(predictions_file)
    
    total_samples = len(df)
    gcn_acc = (df["gcn_correct"].sum() / total_samples) * 100
    gat_acc = (df["gat_correct"].sum() / total_samples) * 100
    agreement = ((df["gcn_pred"] == df["gat_pred"]).sum() / total_samples) * 100
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Evaluated Papers", f"{total_samples:,}")
    m2.metric("GCN Sample Accuracy", f"{gcn_acc:.2f}%")
    m3.metric("GAT Sample Accuracy", f"{gat_acc:.2f}%")
    m4.metric("Model Agreement", f"{agreement:.2f}%")
    
    st.markdown("---")
    
    col_search, col_cat, col_filter = st.columns([1.5, 1.5, 1.5])
    
    with col_search:
        search_query = st.text_input("🔍 Search by Node ID or Paper ID", "").strip()
        
    with col_cat:
        categories = ["All Categories"] + sorted(df["true_label"].unique().tolist())
        selected_cat = st.selectbox("📂 Filter by True Subject", categories)
        
    with col_filter:
        status_options = [
            "All Predictions",
            "Both Correct (GCN & GAT)",
            "Model Disagreement (GCN != GAT)",
            "GCN Correct only",
            "GAT Correct only",
            "Both Incorrect"
        ]
        selected_status = st.selectbox("🎯 Filter by Outcome", status_options)
        
    filtered_df = df.copy()
    
    if search_query:
        if search_query.isdigit():
            val = int(search_query)
            filtered_df = filtered_df[(filtered_df["node_id"] == val) | (filtered_df["paper_id"] == val)]
        else:
            filtered_df = filtered_df[
                filtered_df["true_label"].str.contains(search_query, case=False, na=False) |
                filtered_df["gcn_pred"].str.contains(search_query, case=False, na=False) |
                filtered_df["gat_pred"].str.contains(search_query, case=False, na=False)
            ]
            
    if selected_cat != "All Categories":
        filtered_df = filtered_df[filtered_df["true_label"] == selected_cat]
        
    if selected_status == "Both Correct (GCN & GAT)":
        filtered_df = filtered_df[filtered_df["gcn_correct"] & filtered_df["gat_correct"]]
    elif selected_status == "Model Disagreement (GCN != GAT)":
        filtered_df = filtered_df[filtered_df["gcn_pred"] != filtered_df["gat_pred"]]
    elif selected_status == "GCN Correct only":
        filtered_df = filtered_df[filtered_df["gcn_correct"] & (~filtered_df["gat_correct"])]
    elif selected_status == "GAT Correct only":
        filtered_df = filtered_df[(~filtered_df["gcn_correct"]) & filtered_df["gat_correct"]]
    elif selected_status == "Both Incorrect":
        filtered_df = filtered_df[(~filtered_df["gcn_correct"]) & (~filtered_df["gat_correct"])]
        
    st.caption(f"Showing {len(filtered_df):,} matching papers out of {total_samples:,}")
    
    # Format table for display
    display_df = filtered_df.copy()
    display_df["GCN Status"] = display_df["gcn_correct"].apply(lambda x: "✅ Correct" if x else "❌ Wrong")
    display_df["GAT Status"] = display_df["gat_correct"].apply(lambda x: "✅ Correct" if x else "❌ Wrong")
    
    cols_to_show = ["node_id", "paper_id", "true_label", "gcn_pred", "GCN Status", "gat_pred", "GAT Status"]
    st.dataframe(
        display_df[cols_to_show].rename(columns={
            "node_id": "Node ID",
            "paper_id": "MAG Paper ID",
            "true_label": "Ground Truth Category",
            "gcn_pred": "GCN Prediction",
            "gat_pred": "GAT Prediction"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    if len(filtered_df) > 0:
        with st.expander("🔎 Paper Detail Inspector", expanded=False):
            paper_options = filtered_df["node_id"].head(50).tolist()
            inspected_node = st.selectbox("Select Node ID to inspect", paper_options)
            row = filtered_df[filtered_df["node_id"] == inspected_node].iloc[0]
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.write(f"**Node Index:** `{row['node_id']}`")
                st.write(f"**MAG Paper ID:** `{row['paper_id']}`")
                st.write(f"**True Category:** `{row['true_label']}`")
            with p_col2:
                st.write(f"**GCN Output:** `{row['gcn_pred']}` ({'✅ Correct' if row['gcn_correct'] else '❌ Incorrect'})")
                st.write(f"**GAT Output:** `{row['gat_pred']}` ({'✅ Correct' if row['gat_correct'] else '❌ Incorrect'})")
