from pathlib import Path
import pandas as pd
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


def render_classification_demo(predictions_file: Path | None = None) -> None:
    if predictions_file is None:
        dashboard_dir = Path(__file__).resolve().parent.parent
        predictions_file = dashboard_dir.parent / "results" / "evaluation" / "paper_predictions.csv"

    resolved_pred_file = _resolve_candidate_file("paper_predictions.csv", predictions_file.parent if predictions_file else None)

    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="margin: 0; font-weight: 800; font-size: 1.7rem; color: #FFFFFF; letter-spacing: -0.02em;">
            🔬 Real-Time Paper Classification & Node Lookup
        </h2>
        <p style="color: #94A3B8; margin-top: 5px; font-size: 0.98rem; line-height: 1.5;">
            Search and inspect individual academic paper predictions, model agreements, and classification errors via liquid glass query cockpit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not resolved_pred_file or not resolved_pred_file.exists():
        st.info("💡 Run the evaluation notebook to export paper-level predictions for interactive lookup.")
        return

    df = pd.read_csv(resolved_pred_file)
    total_samples = len(df)
    gcn_correct_cnt = int(df["gcn_correct"].sum())
    gat_correct_cnt = int(df["gat_correct"].sum())
    agreement_cnt = int((df["gcn_pred"] == df["gat_pred"]).sum())

    # Liquid Glass Metrics Summary Cards
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
        <div class="metric-card highlight-gcn">
            <div class="metric-icon">🔹</div>
            <div class="metric-value" style="color: #38BDF8;">{(gcn_correct_cnt/total_samples)*100:.1f}%</div>
            <div class="metric-label">GCN Accuracy</div>
            <div class="metric-sub" style="color: #38BDF8;">{gcn_correct_cnt:,} / {total_samples:,} correct</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card highlight-gat">
            <div class="metric-icon">🔸</div>
            <div class="metric-value" style="color: #C084FC;">{(gat_correct_cnt/total_samples)*100:.1f}%</div>
            <div class="metric-label">GAT Accuracy</div>
            <div class="metric-sub" style="color: #C084FC;">{gat_correct_cnt:,} / {total_samples:,} correct</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🤝</div>
            <div class="metric-value" style="color: #34D399;">{(agreement_cnt/total_samples)*100:.1f}%</div>
            <div class="metric-label">Model Consensus</div>
            <div class="metric-sub" style="color: #34D399;">{agreement_cnt:,} identical predictions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Search & Filter Controls in Liquid Glass Card
    st.markdown("""
    <div class="glass-img-container" style="padding: 18px 20px; margin-bottom: 16px;">
        <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            <span>⚡ Interactive Query & Filter Engine</span>
        </div>
    """, unsafe_allow_html=True)

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
        st.markdown(f"<span style='color: #94A3B8; font-size: 0.92rem;'>Displaying <b style='color: #38BDF8;'>{len(filtered_df):,}</b> matching publications</span>", unsafe_allow_html=True)
    with c_rand:
        if st.button("🎲 Pick Random Paper", use_container_width=True):
            if len(filtered_df) > 0:
                random_pick = int(filtered_df.sample(1)["node_id"].iloc[0])
                st.session_state["selected_inspect_node"] = random_pick

    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Table inside Liquid Glass Container
    display_df = filtered_df.copy()
    display_df["GCN Status"] = display_df["gcn_correct"].apply(lambda x: "✅ Correct" if x else "❌ Wrong")
    display_df["GAT Status"] = display_df["gat_correct"].apply(lambda x: "✅ Correct" if x else "❌ Wrong")
    display_df["Consensus"] = (display_df["gcn_pred"] == display_df["gat_pred"]).apply(lambda x: "🤝 Agree" if x else "⚡ Disagree")

    cols_to_show = ["node_id", "paper_id", "true_label", "gcn_pred", "GCN Status", "gat_pred", "GAT Status", "Consensus"]
    
    st.markdown("""
    <div class="glass-img-container" style="padding: 12px; margin-bottom: 16px;">
    """, unsafe_allow_html=True)
    st.dataframe(
        display_df[cols_to_show].rename(columns={
            "node_id": "Node ID",
            "paper_id": "MAG Paper ID",
            "true_label": "Ground Truth Class",
            "gcn_pred": "GCN Prediction",
            "gat_pred": "GAT Prediction"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Paper Card Inspector in Liquid Glass Cockpit
    if len(filtered_df) > 0:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Deep Paper Inspector & Model Diagnosis Cockpit", expanded=True):
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
                <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(15, 23, 42, 0.85) 100%); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.12); border-top: 1px solid rgba(255,255,255,0.3); box-shadow: 0 10px 25px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.2);">
                    <div style="color: #94A3B8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Paper Metadata</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">Node #{row['node_id']}</div>
                    <div style="color: #38BDF8; font-size: 0.9rem; margin-top: 4px;">MAG ID: <code>{row['paper_id']}</code></div>
                    <div style="margin-top: 14px; padding: 10px 14px; background: rgba(56, 189, 248, 0.12); border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.3);">
                        <div style="color: #94A3B8; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">GROUND TRUTH CATEGORY</div>
                        <div style="color: #38BDF8; font-weight: 800; font-size: 1.08rem; margin-top: 2px;">{true_cat}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_p2:
                gcn_status_badge = "background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981;" if gcn_is_corr else "background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444;"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(15, 23, 42, 0.85) 100%); padding: 20px; border-radius: 16px; border: 1px solid rgba(56, 189, 248, 0.35); border-top: 1px solid rgba(255, 255, 255, 0.35); box-shadow: 0 10px 25px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #38BDF8; font-weight: 800; font-size: 1.1rem;">🔹 GCN Output</span>
                        <span style="padding: 4px 10px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; {gcn_status_badge}">
                            {'MATCH' if gcn_is_corr else 'MISMATCH'}
                        </span>
                    </div>
                    <div style="margin-top: 14px;">
                        <div style="color: #94A3B8; font-size: 0.78rem; font-weight: 600;">Predicted Subject:</div>
                        <div style="font-size: 1.18rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">{gcn_pred_cat}</div>
                    </div>
                    <div style="margin-top: 12px; font-size: 0.86rem; font-weight: 600; color: {'#34D399' if gcn_is_corr else '#F87171'};">
                        {gcn_detail_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_p3:
                gat_status_badge = "background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981;" if gat_is_corr else "background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444;"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(192, 132, 252, 0.1) 0%, rgba(15, 23, 42, 0.85) 100%); padding: 20px; border-radius: 16px; border: 1px solid rgba(192, 132, 252, 0.35); border-top: 1px solid rgba(255, 255, 255, 0.35); box-shadow: 0 10px 25px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #C084FC; font-weight: 800; font-size: 1.1rem;">🔸 GAT Output</span>
                        <span style="padding: 4px 10px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; {gat_status_badge}">
                            {'MATCH' if gat_is_corr else 'MISMATCH'}
                        </span>
                    </div>
                    <div style="margin-top: 14px;">
                        <div style="color: #94A3B8; font-size: 0.78rem; font-weight: 600;">Predicted Subject:</div>
                        <div style="font-size: 1.18rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">{gat_pred_cat}</div>
                    </div>
                    <div style="margin-top: 12px; font-size: 0.86rem; font-weight: 600; color: {'#34D399' if gat_is_corr else '#F87171'};">
                        {gat_detail_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
