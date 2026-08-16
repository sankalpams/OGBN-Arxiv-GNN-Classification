from pathlib import Path
import pandas as pd
import streamlit as st


def render_classification_demo(predictions_file: Path | None = None) -> None:
    if predictions_file is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        predictions_file = dashboard_dir.parent / "results" / "evaluation" / "paper_predictions.csv"

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; font-weight: 700; font-size: 1.6rem; color: #F8FAFC;">
            🔬 Real-Time Paper Classification & Node Lookup
        </h2>
        <p style="color: #94A3B8; margin-top: 4px; font-size: 0.95rem;">
            Search and inspect individual academic paper predictions, model agreements, and classification errors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not predictions_file.exists():
        st.info("💡 Run the evaluation notebook to export paper-level predictions for interactive lookup.")
        return

    df = pd.read_csv(predictions_file)
    total_samples = len(df)
    gcn_correct_cnt = int(df["gcn_correct"].sum())
    gat_correct_cnt = int(df["gat_correct"].sum())
    agreement_cnt = int((df["gcn_pred"] == df["gat_pred"]).sum())

    # Metrics Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📑</div>
            <div class="metric-value">{total_samples:,}</div>
            <div class="metric-label">Evaluated Sample Papers</div>
            <div class="metric-sub">Official Test Partition</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔹</div>
            <div class="metric-value" style="color: #38BDF8;">{(gcn_correct_cnt/total_samples)*100:.1f}%</div>
            <div class="metric-label">GCN Accuracy</div>
            <div class="metric-sub">{gcn_correct_cnt:,} / {total_samples:,} correct</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔸</div>
            <div class="metric-value" style="color: #C084FC;">{(gat_correct_cnt/total_samples)*100:.1f}%</div>
            <div class="metric-label">GAT Accuracy</div>
            <div class="metric-sub">{gat_correct_cnt:,} / {total_samples:,} correct</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🤝</div>
            <div class="metric-value" style="color: #10B981;">{(agreement_cnt/total_samples)*100:.1f}%</div>
            <div class="metric-label">Model Consensus</div>
            <div class="metric-sub">{agreement_cnt:,} identical predictions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Search & Filter Controls
    col_search, col_cat, col_filter = st.columns([1.8, 1.4, 1.4])

    with col_search:
        search_query = st.text_input("🔍 Search by Node ID, MAG Paper ID, or Subject Keyword", "").strip()

    with col_cat:
        categories = ["All Categories"] + sorted(df["true_label"].dropna().unique().tolist())
        selected_cat = st.selectbox("📂 Filter by Ground Truth Subject", categories)

    with col_filter:
        status_options = [
            "All Predictions",
            "Both Correct (GCN & GAT)",
            "Model Disagreement (GCN != GAT)",
            "GCN Correct Only",
            "GAT Correct Only",
            "Both Incorrect"
        ]
        selected_status = st.selectbox("🎯 Filter by Outcome", status_options)

    # Filter Logic
    filtered_df = df.copy()

    if search_query:
        if search_query.isdigit():
            val = int(search_query)
            filtered_df = filtered_df[(filtered_df["node_id"] == val) | (filtered_df["paper_id"] == val)]
        else:
            filtered_df = filtered_df[
                filtered_df["true_label"].astype(str).str.contains(search_query, case=False, na=False) |
                filtered_df["gcn_pred"].astype(str).str.contains(search_query, case=False, na=False) |
                filtered_df["gat_pred"].astype(str).str.contains(search_query, case=False, na=False)
            ]

    if selected_cat != "All Categories":
        filtered_df = filtered_df[filtered_df["true_label"] == selected_cat]

    if selected_status == "Both Correct (GCN & GAT)":
        filtered_df = filtered_df[filtered_df["gcn_correct"] & filtered_df["gat_correct"]]
    elif selected_status == "Model Disagreement (GCN != GAT)":
        filtered_df = filtered_df[filtered_df["gcn_pred"] != filtered_df["gat_pred"]]
    elif selected_status == "GCN Correct Only":
        filtered_df = filtered_df[filtered_df["gcn_correct"] & (~filtered_df["gat_correct"])]
    elif selected_status == "GAT Correct Only":
        filtered_df = filtered_df[(~filtered_df["gcn_correct"]) & filtered_df["gat_correct"]]
    elif selected_status == "Both Incorrect":
        filtered_df = filtered_df[(~filtered_df["gcn_correct"]) & (~filtered_df["gat_correct"])]

    # Result Count & Quick Randomizer
    c_count, c_rand = st.columns([3, 1])
    with c_count:
        st.markdown(f"<span style='color: #94A3B8; font-size: 0.9rem;'>Displaying <b style='color: #38BDF8;'>{len(filtered_df):,}</b> matching publications</span>", unsafe_allow_html=True)
    with c_rand:
        if st.button("🎲 Pick Random Paper", width="stretch"):
            if len(filtered_df) > 0:
                random_pick = int(filtered_df.sample(1)["node_id"].iloc[0])
                st.session_state["selected_inspect_node"] = random_pick

    # Interactive Table
    display_df = filtered_df.copy()
    display_df["GCN Status"] = display_df["gcn_correct"].apply(lambda x: "✅ Correct" if x else "❌ Wrong")
    display_df["GAT Status"] = display_df["gat_correct"].apply(lambda x: "✅ Correct" if x else "❌ Wrong")
    display_df["Consensus"] = (display_df["gcn_pred"] == display_df["gat_pred"]).apply(lambda x: "🤝 Agree" if x else "⚡ Disagree")

    cols_to_show = ["node_id", "paper_id", "true_label", "gcn_pred", "GCN Status", "gat_pred", "GAT Status", "Consensus"]
    st.dataframe(
        display_df[cols_to_show].rename(columns={
            "node_id": "Node ID",
            "paper_id": "MAG Paper ID",
            "true_label": "Ground Truth Class",
            "gcn_pred": "GCN Prediction",
            "gat_pred": "GAT Prediction"
        }),
        width="stretch",
        hide_index=True
    )

    # Detailed Paper Card Inspector
    if len(filtered_df) > 0:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Deep Paper Inspector & Model Diagnosis", expanded=True):
            node_options = filtered_df["node_id"].tolist()
            
            default_index = 0
            if "selected_inspect_node" in st.session_state and st.session_state["selected_inspect_node"] in node_options:
                default_index = node_options.index(st.session_state["selected_inspect_node"])
                
            selected_node = st.selectbox("Select Paper Node ID to inspect in detail:", node_options, index=default_index)
            row = filtered_df[filtered_df["node_id"] == selected_node].iloc[0]

            true_cat = str(row['true_label'])
            gcn_pred_cat = str(row['gcn_pred'])
            gat_pred_cat = str(row['gat_pred'])
            gcn_is_corr = bool(row['gcn_correct'])
            gat_is_corr = bool(row['gat_correct'])

            gcn_detail_msg = "✔ Successfully predicted ground truth" if gcn_is_corr else f"✘ Expected {true_cat}"
            gat_detail_msg = "✔ Successfully predicted ground truth" if gat_is_corr else f"✘ Expected {true_cat}"

            col_p1, col_p2, col_p3 = st.columns([1.2, 1.4, 1.4])
            
            with col_p1:
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); padding: 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                    <div style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;">Paper Metadata</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Node #{row['node_id']}</div>
                    <div style="color: #38BDF8; font-size: 0.9rem; margin-top: 4px;">MAG ID: <code>{row['paper_id']}</code></div>
                    <div style="margin-top: 12px; padding: 8px 12px; background: rgba(56, 189, 248, 0.1); border-radius: 8px;">
                        <div style="color: #94A3B8; font-size: 0.75rem;">GROUND TRUTH CATEGORY</div>
                        <div style="color: #38BDF8; font-weight: 700; font-size: 1.05rem;">{true_cat}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_p2:
                gcn_status_badge = "background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981;" if gcn_is_corr else "background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid #EF4444;"
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); padding: 18px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #38BDF8; font-weight: 700; font-size: 1.05rem;">🔹 GCN Output</span>
                        <span style="padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; {gcn_status_badge}">
                            {'MATCH' if gcn_is_corr else 'MISMATCH'}
                        </span>
                    </div>
                    <div style="margin-top: 12px;">
                        <div style="color: #94A3B8; font-size: 0.8rem;">Predicted Subject:</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">{gcn_pred_cat}</div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.85rem; color: {'#10B981' if gcn_is_corr else '#EF4444'};">
                        {gcn_detail_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_p3:
                gat_status_badge = "background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981;" if gat_is_corr else "background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid #EF4444;"
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); padding: 18px; border-radius: 12px; border: 1px solid rgba(192, 132, 252, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #C084FC; font-weight: 700; font-size: 1.05rem;">🔸 GAT Output</span>
                        <span style="padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; {gat_status_badge}">
                            {'MATCH' if gat_is_corr else 'MISMATCH'}
                        </span>
                    </div>
                    <div style="margin-top: 12px;">
                        <div style="color: #94A3B8; font-size: 0.8rem;">Predicted Subject:</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">{gat_pred_cat}</div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.85rem; color: {'#10B981' if gat_is_corr else '#EF4444'};">
                        {gat_detail_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
