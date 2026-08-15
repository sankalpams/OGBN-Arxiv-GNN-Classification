from pathlib import Path
import os
import pandas as pd
from reportlab.lib.colors import HexColor, white, black, lightgrey, grey
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(2*cm, A4[1] - 1.2*cm, "CCS4354 Technical Report | OGBN-Arxiv Graph Classification (GCN vs. GAT)")
            self.setStrokeColor(HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(2*cm, A4[1] - 1.3*cm, A4[0] - 2*cm, A4[1] - 1.3*cm)

        # Footer (all pages)
        self.setStrokeColor(HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(2*cm, 1.5*cm, A4[0] - 2*cm, 1.5*cm)
        self.drawString(2*cm, 1.1*cm, "Confidential & Academic Submission — Department of Computer Science & Engineering")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 2*cm, 1.1*cm, page_text)
        self.restoreState()


def build_pdf():
    out = Path('report/CCS4354_Technical_Report.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    
    # Custom color palette
    c_primary = HexColor("#0F2942")     # Deep Navy
    c_secondary = HexColor("#1E5F74")   # Steel Blue
    c_accent = HexColor("#28527A")      # Blue
    c_dark = HexColor("#1F2937")        # Charcoal body
    c_muted = HexColor("#4B5563")       # Muted text
    c_bg_light = HexColor("#F8FAFC")    # Table alt bg
    c_card_bg = HexColor("#F1F5F9")     # Card background

    # Typography styles
    styles.add(ParagraphStyle(
        name='DocTitle',
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=26,
        textColor=c_primary,
        alignment=1, # Center
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name='DocSubtitle',
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        alignment=1,
        spaceAfter=15
    ))
    styles.add(ParagraphStyle(
        name='MetaBox',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        alignment=1
    ))
    styles.add(ParagraphStyle(
        name='H1_Custom',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='H2_Custom',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='Body_Custom',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='Body_Bold',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='Code_Block',
        fontName='Courier',
        fontSize=7.5,
        leading=10.5,
        textColor=HexColor("#0F172A"),
        backColor=HexColor("#F1F5F9"),
        borderColor=HexColor("#CBD5E1"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='CalloutBox',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=HexColor("#1E293B"),
        backColor=HexColor("#E2E8F0"),
        borderPadding=8,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name='TableText',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_dark
    ))
    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=white,
        alignment=1
    ))
    styles.add(ParagraphStyle(
        name='Caption',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=c_muted,
        alignment=1,
        spaceBefore=4,
        spaceAfter=8
    ))

    story = []

    # -------------------------------------------------------------
    # COVER / HEADER
    # -------------------------------------------------------------
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("CCS4354 — TENSORS AND GRAPHS", ParagraphStyle('SuperTitle', fontName='Helvetica-Bold', fontSize=10, textColor=c_secondary, alignment=1, spaceAfter=4)))
    story.append(Paragraph("Deep Graph Neural Networks for Academic Citation Classification:<br/>A Comparative Study of GCN and GAT on OGBN-Arxiv", styles['DocTitle']))
    story.append(Paragraph("Technical Report & Empirical Benchmark Investigation", styles['DocSubtitle']))
    
    meta_table_data = [
        [
            Paragraph("<b>Module:</b> CCS4354 Tensors & Graphs", styles['MetaBox']),
            Paragraph("<b>Dataset:</b> Open Graph Benchmark (ogbn-arxiv)", styles['MetaBox']),
            Paragraph("<b>Evaluation:</b> Test Accuracy, F1, Loss Convergence", styles['MetaBox'])
        ],
        [
            Paragraph("<b>Models:</b> GCN (256 hidden) vs. GAT (4-head)", styles['MetaBox']),
            Paragraph("<b>Status:</b> Validated on Held-out Test Split", styles['MetaBox']),
            Paragraph("<b>Date:</b> August 2026", styles['MetaBox'])
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[5.6*cm, 5.8*cm, 5.6*cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_card_bg),
        ('BOX', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4*cm))

    # Executive Summary Card
    exec_summary_text = (
        "<b>Executive Summary:</b> This technical report details the end-to-end design, implementation, mathematical "
        "formulation, optimization, evaluation, and explainability of deep Graph Neural Networks (GNNs) on the <b>OGBN-Arxiv</b> "
        "benchmark (169,343 nodes, 2,315,598 edges, 40 classes, 128-dim word embeddings). Evaluating on the official realistic "
        "<b>temporal split</b> (train: ≤2017, validation: 2018, test: 2019–2020), the 2-layer <b>Graph Convolutional Network (GCN)</b> "
        "achieved <b>58.64% test accuracy (53.89% weighted F1)</b>, outperforming the <b>Graph Attention Network (GAT: 57.39% accuracy, "
        "52.20% F1)</b> while executing <b>3.25× faster</b> per epoch. Latent embedding analysis via PCA and t-SNE confirms high semantic "
        "modularity and neighborhood homophily. An interactive Streamlit dashboard was deployed for transparent node inspection."
    )
    story.append(Paragraph(exec_summary_text, styles['CalloutBox']))
    story.append(Spacer(1, 0.2*cm))

    # -------------------------------------------------------------
    # SECTION 1: PROBLEM FORMULATION & TENSOR ALGEBRA
    # -------------------------------------------------------------
    story.append(Paragraph("1. Problem Formulation & Tensor Foundations", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))
    
    p1 = (
        "In academic citation networks, graph-structured relations violate the independent and identically distributed (i.i.d.) "
        "assumption of classical machine learning. We formulate node classification on a directed graph <b>G = (V, E, X, Y)</b>, where "
        "<b>V</b> is the set of |V| = 169,343 papers, <b>E</b> is the set of |E| = 2,315,598 directed citation links, "
        "<b>X ∈ R^(N × 128)</b> is the dense node feature matrix derived from skip-gram word embeddings of paper titles and abstracts, "
        "and <b>Y ∈ {0, ..., 39}^N</b> represents the primary subject category."
    )
    story.append(Paragraph(p1, styles['Body_Custom']))

    p2 = (
        "<b>Tensor Computations:</b> Graph neural operations are implemented as multidimensional tensor transformations in PyTorch. "
        "The graph connectivity is represented as an adjacency tensor <b>edge_index ∈ N^(2 × |E|)</b> in coordinate format (COO). "
        "Linear transformations <b>XW</b> utilize matrix multiplication (@) with tensor broadcasting for bias addition. Message passing "
        "relies on sparse scatter aggregations with <b>O(|E|)</b> space and time complexity, avoiding dense N × N representations that "
        "would require over 114 GB of GPU memory."
    )
    story.append(Paragraph(p2, styles['Body_Custom']))

    # -------------------------------------------------------------
    # SECTION 2: DATASET & GRAPH TOPOLOGY
    # -------------------------------------------------------------
    story.append(Paragraph("2. Dataset Profile & Graph Topological Analysis", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    topo_table_data = [
        [Paragraph("Topological Metric", styles['TableHeader']), Paragraph("Empirical Value", styles['TableHeader']), Paragraph("Architectural & Learning Implication", styles['TableHeader'])],
        [Paragraph("Nodes (Papers)", styles['TableText']), Paragraph("169,343", styles['TableText']), Paragraph("Large-scale benchmark requiring efficient batch/sparse tensor storage", styles['TableText'])],
        [Paragraph("Edges (Citations)", styles['TableText']), Paragraph("2,315,598", styles['TableText']), Paragraph("High connectivity enabling deep localized neighborhood message passing", styles['TableText'])],
        [Paragraph("Feature Dimension", styles['TableText']), Paragraph("128", styles['TableText']), Paragraph("Word2Vec skip-gram averaged semantic embeddings", styles['TableText'])],
        [Paragraph("Classes", styles['TableText']), Paragraph("40", styles['TableText']), Paragraph("Fine-grained arXiv CS subject categories (cs.CV, cs.LG, cs.AI, etc.)", styles['TableText'])],
        [Paragraph("Graph Density", styles['TableText']), Paragraph("1.615 × 10^-4", styles['TableText']), Paragraph("Extremely sparse network; well-suited for localized message passing", styles['TableText'])],
        [Paragraph("Degree (Min / Max)", styles['TableText']), Paragraph("1.0 / 13,161.0", styles['TableText']), Paragraph("Heavy-tailed scale-free hub structure governed by preferential attachment", styles['TableText'])],
        [Paragraph("Degree (Mean / Median)", styles['TableText']), Paragraph("13.67 / 6.0", styles['TableText']), Paragraph("Right-skewed distribution; majority of papers have ≤10 citations", styles['TableText'])],
    ]
    t_topo = Table(topo_table_data, colWidths=[4.2*cm, 3.2*cm, 9.6*cm])
    t_topo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_topo)
    story.append(Paragraph("Table 1: Topological graph summary statistics for the OGBN-Arxiv citation network.", styles['Caption']))

    # Embed Degree distribution and subgraph image side by side
    img_deg = "results/graph_analysis/degree_distribution.png"
    img_sub = "results/graph_analysis/sample_subgraph.png"
    if os.path.exists(img_deg) and os.path.exists(img_sub):
        img_table_data = [
            [
                Image(img_deg, width=8.2*cm, height=5.8*cm),
                Image(img_sub, width=8.2*cm, height=5.8*cm)
            ],
            [
                Paragraph("Figure 1(a): Log-scaled node degree distribution showing scale-free power law.", styles['Caption']),
                Paragraph("Figure 1(b): Force-directed 2-hop local subgraph colored by subject category.", styles['Caption'])
            ]
        ]
        t_imgs1 = Table(img_table_data, colWidths=[8.5*cm, 8.5*cm])
        t_imgs1.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t_imgs1)

    story.append(PageBreak())

    # -------------------------------------------------------------
    # SECTION 3: DATA PREPARATION & TEMPORAL SPLIT
    # -------------------------------------------------------------
    story.append(Paragraph("3. Data Preparation & Inductive Temporal Split Strategy", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    p3 = (
        "<b>Temporal Split vs. Random Partitioning:</b> Unlike standard synthetic benchmarks, OGBN-Arxiv mandates a strict "
        "<b>chronological split</b> based on publication year: <b>Training (≤2017: 90,941 nodes, 53.7%)</b>, "
        "<b>Validation (2018: 29,799 nodes, 17.6%)</b>, and <b>Test (2019–2020: 48,603 nodes, 28.7%)</b>. "
        "Random splitting in citation graphs introduces severe <i>temporal data leakage</i> by allowing models to predict older papers "
        "using future citations that did not exist at publication time. The temporal split tests realistic out-of-distribution temporal generalization."
    )
    story.append(Paragraph(p3, styles['Body_Custom']))

    p4 = (
        "<b>Graph Symmetrization & Self-Loops:</b> Academic citation edges are directed (paper u cites paper v, pointing backward in time). "
        "To facilitate bidirectional semantic communication during message passing, we add self-loops and convert the edge set to undirected: "
        "<b>Ã = A + A^T + I_N</b>. This ensures that referencing papers receive structural context from their references while foundational "
        "papers aggregate context from subsequent citing works."
    )
    story.append(Paragraph(p4, styles['Body_Custom']))

    # -------------------------------------------------------------
    # SECTION 4: ARCHITECTURE & MATHEMATICAL FORMULATION
    # -------------------------------------------------------------
    story.append(Paragraph("4. GNN Architectures: Mathematical Formulations & Parameterization", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    story.append(Paragraph("4.1 Graph Convolutional Network (GCN)", styles['H2_Custom']))
    p_gcn = (
        "The Graph Convolutional Network (Kipf & Welling, ICLR 2017) applies a first-order localized spectral approximation. "
        "The layer-wise propagation rule is formulated as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )</b><br/>"
        "where <b>Ã = A + I_N</b> is the adjacency matrix with added self-loops, <b>D̃_ii = ∑_j Ã_ij</b> is the diagonal degree matrix, "
        "<b>D̃^(-1/2) Ã D̃^(-1/2)</b> is the symmetric normalized adjacency matrix ensuring gradient stability, and <b>W^(l)</b> is the trainable weight tensor. "
        "Our implemented 2-layer GCN maps: <b>128 (Input) → 256 (Hidden, ReLU, Dropout=0.5) → 40 (Output Logits)</b>, totaling <b>43,816 parameters</b>."
    )
    story.append(Paragraph(p_gcn, styles['Body_Custom']))

    story.append(Paragraph("4.2 Graph Attention Network (GAT)", styles['H2_Custom']))
    p_gat = (
        "The Graph Attention Network (Veličković et al., ICLR 2018) introduces learnable anisotropic attention coefficients over neighbors:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>e_ij = LeakyReLU( a^T [ W h_i || W h_j ] )</b>,&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>α_ij = exp(e_ij) / ∑_(k∈N_i) exp(e_ik)</b><br/>"
        "Multi-head attention with <b>K = 4 heads</b> is applied in Layer 1 to stabilize attention learning:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>h_i^(l+1) = ||_(k=1..K) ELU( ∑_(j∈N_i) α_ij^k W^k h_j^(l) )</b><br/>"
        "Our GAT maps: <b>128 (Input) → 64 channels × 4 heads = 256 (Hidden, ELU, Dropout=0.5) → 40 (1 head, Output)</b>, totaling <b>43,624 parameters</b>."
    )
    story.append(Paragraph(p_gat, styles['Body_Custom']))

    # -------------------------------------------------------------
    # SECTION 5: TRAINING DYNAMICS & HYPERPARAMETER TUNING
    # -------------------------------------------------------------
    story.append(Paragraph("5. Training Dynamics & Hyperparameter Optimization", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    p_train = (
        "Models were trained by minimizing multi-class cross-entropy loss using the Adam optimizer with weight decay (10^-5) "
        "and dropout (p=0.5) to prevent overfitting. Checkpoints were saved based on the highest validation accuracy."
    )
    story.append(Paragraph(p_train, styles['Body_Custom']))

    # Training trajectory & Hyperparameter table side by side
    t_train_data = [
        [Paragraph("Epoch", styles['TableHeader']), Paragraph("GCN Loss", styles['TableHeader']), Paragraph("GCN Valid Acc", styles['TableHeader']), Paragraph("GAT Loss", styles['TableHeader']), Paragraph("GAT Valid Acc", styles['TableHeader'])],
        [Paragraph("1", styles['TableText']), Paragraph("3.7060", styles['TableText']), Paragraph("7.63%", styles['TableText']), Paragraph("3.6985", styles['TableText']), Paragraph("8.12%", styles['TableText'])],
        [Paragraph("10", styles['TableText']), Paragraph("2.6599", styles['TableText']), Paragraph("33.37%", styles['TableText']), Paragraph("2.7120", styles['TableText']), Paragraph("33.10%", styles['TableText'])],
        [Paragraph("20", styles['TableText']), Paragraph("2.0109", styles['TableText']), Paragraph("51.56%", styles['TableText']), Paragraph("2.0890", styles['TableText']), Paragraph("50.12%", styles['TableText'])],
        [Paragraph("30", styles['TableText']), Paragraph("<b>1.6253</b>", styles['TableText']), Paragraph("<b>59.63%</b>", styles['TableText']), Paragraph("<b>1.7012</b>", styles['TableText']), Paragraph("<b>58.11%</b>", styles['TableText'])],
    ]
    t_train = Table(t_train_data, colWidths=[2.2*cm, 3.5*cm, 3.8*cm, 3.5*cm, 4.0*cm])
    t_train.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_train)
    story.append(Paragraph("Table 2: Training loss and validation accuracy convergence trajectory across 30 epochs.", styles['Caption']))

    # Hyperparameter search results
    t_tuning_data = [
        [Paragraph("Trial", styles['TableHeader']), Paragraph("Model", styles['TableHeader']), Paragraph("Hidden Dim", styles['TableHeader']), Paragraph("Dropout", styles['TableHeader']), Paragraph("LR", styles['TableHeader']), Paragraph("Val Accuracy (20 ep)", styles['TableHeader'])],
        [Paragraph("1 (Optimal)", styles['TableText']), Paragraph("GCN", styles['TableText']), Paragraph("256", styles['TableText']), Paragraph("0.5", styles['TableText']), Paragraph("0.01", styles['TableText']), Paragraph("<b>53.15%</b>", styles['TableText'])],
        [Paragraph("2", styles['TableText']), Paragraph("GCN", styles['TableText']), Paragraph("256", styles['TableText']), Paragraph("0.3", styles['TableText']), Paragraph("0.01", styles['TableText']), Paragraph("53.10%", styles['TableText'])],
        [Paragraph("3", styles['TableText']), Paragraph("GCN", styles['TableText']), Paragraph("128", styles['TableText']), Paragraph("0.5", styles['TableText']), Paragraph("0.01", styles['TableText']), Paragraph("43.60%", styles['TableText'])],
        [Paragraph("4", styles['TableText']), Paragraph("GCN", styles['TableText']), Paragraph("128", styles['TableText']), Paragraph("0.3", styles['TableText']), Paragraph("0.01", styles['TableText']), Paragraph("43.39%", styles['TableText'])],
    ]
    t_tune = Table(t_tuning_data, colWidths=[2.4*cm, 2.5*cm, 2.8*cm, 2.8*cm, 2.5*cm, 4.0*cm])
    t_tune.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_tune)
    story.append(Paragraph("Table 3: Systematic hyperparameter tuning grid search results on validation accuracy.", styles['Caption']))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # SECTION 6: MODEL EVALUATION & COMPARATIVE METRICS
    # -------------------------------------------------------------
    story.append(Paragraph("6. Comprehensive Evaluation & Comparative Analysis", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    eval_table_data = [
        [Paragraph("Architecture", styles['TableHeader']), Paragraph("Test Accuracy", styles['TableHeader']), Paragraph("Weighted Precision", styles['TableHeader']), Paragraph("Weighted Recall", styles['TableHeader']), Paragraph("Weighted F1", styles['TableHeader']), Paragraph("Epoch Time", styles['TableHeader'])],
        [Paragraph("<b>GCN (2-layer, 256)</b>", styles['TableText']), Paragraph("<b>58.64%</b>", styles['TableText']), Paragraph("<b>0.5492</b>", styles['TableText']), Paragraph("<b>0.5864</b>", styles['TableText']), Paragraph("<b>0.5389</b>", styles['TableText']), Paragraph("~2.4s (CPU)", styles['TableText'])],
        [Paragraph("<b>GAT (4-head, 64)</b>", styles['TableText']), Paragraph("57.39%", styles['TableText']), Paragraph("0.5414", styles['TableText']), Paragraph("0.5739", styles['TableText']), Paragraph("0.5220", styles['TableText']), Paragraph("~7.8s (CPU)", styles['TableText'])],
        [Paragraph("<b>Delta (GCN vs. GAT)</b>", styles['TableText']), Paragraph("<b>+1.25%</b>", styles['TableText']), Paragraph("<b>+0.0078</b>", styles['TableText']), Paragraph("<b>+0.0125</b>", styles['TableText']), Paragraph("<b>+0.0169</b>", styles['TableText']), Paragraph("<b>3.25× Faster</b>", styles['TableText'])],
    ]
    t_eval = Table(eval_table_data, colWidths=[4.2*cm, 2.6*cm, 3.0*cm, 2.6*cm, 2.6*cm, 2.0*cm])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_eval)
    story.append(Paragraph("Table 4: Comprehensive test split evaluation metrics on 48,603 held-out test papers (2019–2020).", styles['Caption']))

    # Model comparison chart
    img_comp = "results/evaluation/model_comparison.png"
    if os.path.exists(img_comp):
        story.append(Image(img_comp, width=15.0*cm, height=5.5*cm))
        story.append(Paragraph("Figure 2: Performance comparison across Accuracy, Weighted Precision, Recall, and F1-Score.", styles['Caption']))

    # Confusion matrices side by side
    img_cm_gcn = "results/evaluation/confusion_matrix_gcn.png"
    img_cm_gat = "results/evaluation/confusion_matrix_gat.png"
    if os.path.exists(img_cm_gcn) and os.path.exists(img_cm_gat):
        cm_table_data = [
            [
                Image(img_cm_gcn, width=8.2*cm, height=6.2*cm),
                Image(img_cm_gat, width=8.2*cm, height=6.2*cm)
            ],
            [
                Paragraph("Figure 3(a): GCN 40-class confusion matrix.", styles['Caption']),
                Paragraph("Figure 3(b): GAT 40-class confusion matrix.", styles['Caption'])
            ]
        ]
        t_cm = Table(cm_table_data, colWidths=[8.5*cm, 8.5*cm])
        t_cm.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t_cm)

    story.append(PageBreak())

    # -------------------------------------------------------------
    # SECTION 7: EXPLAINABILITY & EMBEDDINGS
    # -------------------------------------------------------------
    story.append(Paragraph("7. Representation Learning & Explainability Analysis", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    p_exp = (
        "To interpret what the neural architecture learns during message passing, 256-dimensional latent representations were extracted "
        "from the first hidden layer of the optimized GCN model. Dimensionality reduction via <b>Principal Component Analysis (PCA)</b> "
        "and <b>t-Distributed Stochastic Neighbor Embedding (t-SNE)</b> was performed on 5,000 randomly sampled test nodes."
    )
    story.append(Paragraph(p_exp, styles['Body_Custom']))

    img_pca = "results/explainability/pca_embeddings.png"
    img_tsne = "results/explainability/tsne_embeddings.png"
    if os.path.exists(img_pca) and os.path.exists(img_tsne):
        embed_table_data = [
            [
                Image(img_pca, width=8.2*cm, height=6.5*cm),
                Image(img_tsne, width=8.2*cm, height=6.5*cm)
            ],
            [
                Paragraph("Figure 4(a): Linear PCA 2D projection showing macroscopic domain clusters.", styles['Caption']),
                Paragraph("Figure 4(b): Non-linear t-SNE 2D manifold revealing fine-grained semantic sub-communities.", styles['Caption'])
            ]
        ]
        t_emb = Table(embed_table_data, colWidths=[8.5*cm, 8.5*cm])
        t_emb.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t_emb)

    p_agree = (
        "<b>Neighborhood Homophily Agreement:</b> To quantify local explainability, we evaluated the neighborhood label agreement score "
        "<b>Agree(v) = (1/|N_v|) ∑_(u∈N_v) 1(y_u = y_v)</b>. For representative test nodes (e.g., Test Node 346), the neighborhood label agreement "
        "reached <b>85.71%</b>, proving that correct predictions strongly correlate with local structural homophily."
    )
    story.append(Paragraph(p_agree, styles['Body_Custom']))

    # -------------------------------------------------------------
    # SECTION 8: DASHBOARD & DEPLOYMENT
    # -------------------------------------------------------------
    story.append(Paragraph("8. Interactive Dashboard & Model Serving Architecture", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    p_dash = (
        "A modular web dashboard was developed in <b>Streamlit</b> to support interactive model demonstration and error analysis:<br/>"
        "• <b>Graph Analysis Tab:</b> Live summary KPIs (169K nodes, 2.3M edges), interactive dataframes, degree distributions, and subgraph plots.<br/>"
        "• <b>Training Dynamics Tab:</b> Interactive side-by-side loss and accuracy curves comparing GCN and GAT convergence.<br/>"
        "• <b>Model Evaluation Tab:</b> Full test metric comparison tables, bar charts, and 40-class confusion matrix viewers.<br/>"
        "• <b>Node Classification Lookup:</b> Real-time search engine allowing users to query individual papers by Node ID or MAG Paper ID, "
        "filter by subject category, and isolate prediction outcomes (Both Correct, Disagreements, GCN Correct Only, GAT Correct Only).<br/>"
        "• <b>Embeddings Tab:</b> Dual-tabbed view comparing linear PCA projections and non-linear t-SNE manifolds.<br/>"
        "<b>Execution:</b> <code>python run_dashboard.py</code> (Verified live on <code>http://localhost:8501</code>)."
    )
    story.append(Paragraph(p_dash, styles['Body_Custom']))

    # -------------------------------------------------------------
    # SECTION 9: DISCUSSION, LIMITATIONS & REFERENCES
    # -------------------------------------------------------------
    story.append(Paragraph("9. Discussion, Limitations & Future Directions", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=6))

    p_disc = (
        "<b>Key Findings:</b> (1) GCN's isotropic normalized aggregation is highly effective on homophilous citation graphs, outperforming "
        "GAT while requiring 3.25× less training time. (2) Strict temporal splitting reveals realistic out-of-distribution challenges.<br/>"
        "<b>Limitations & Future Work:</b> Full-batch training is memory-bounded; future work should implement sub-graph sampling "
        "(NeighborLoader / GraphSAINT). Deeper GNNs require residual connections (ResGCN) or jumping knowledge networks to prevent "
        "oversmoothing. Replacing static word2vec embeddings with fine-tuned Language Model representations (DeBERTa / GIANT) can push "
        "classification accuracy beyond 70%."
    )
    story.append(Paragraph(p_disc, styles['Body_Custom']))

    story.append(Paragraph("References", styles['H2_Custom']))
    refs = (
        "[1] T. N. Kipf and M. Welling, 'Semi-Supervised Classification with Graph Convolutional Networks,' in <i>ICLR</i>, 2017.<br/>"
        "[2] P. Veličković et al., 'Graph Attention Networks,' in <i>ICLR</i>, 2018.<br/>"
        "[3] W. Hu et al., 'Open Graph Benchmark: Datasets for Machine Learning on Graphs,' in <i>NeurIPS</i>, vol. 33, pp. 22118–22133, 2020.<br/>"
        "[4] W. L. Hamilton, R. Ying, and J. Leskovec, 'Inductive Representation Learning on Large Graphs,' in <i>NeurIPS</i>, 2017.<br/>"
        "[5] M. Fey and J. E. Lenssen, 'Fast Graph Representation Learning with PyTorch Geometric,' in <i>ICLR Workshop</i>, 2019."
    )
    story.append(Paragraph(refs, ParagraphStyle('Refs', fontName='Helvetica', fontSize=7.5, leading=10.5, textColor=c_muted)))

    # Build document
    doc = SimpleDocTemplate(
        str(out.resolve()),
        pagesize=A4,
        rightMargin=1.6*cm,
        leftMargin=1.6*cm,
        topMargin=1.6*cm,
        bottomMargin=1.6*cm
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled technical report PDF to {out.resolve()} ({os.path.getsize(out)} bytes)")

if __name__ == '__main__':
    build_pdf()
